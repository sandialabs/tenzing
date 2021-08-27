#include "sched/schedule.hpp"

#include "sched/macro_at.hpp"
#include "sched/numeric.hpp"
#include "sched/ops_cuda.hpp"

#include <algorithm>
#include <typeinfo>
#include <typeindex>
#include <numeric>

using BenchResult = Schedule::BenchResult;

int Schedule::remove_redundant_syncs() {

    using node_t = std::shared_ptr<CpuNode>;

    int removed = 0;

    // remove redundant stream syncs
    {
        bool changed = true;
        while(changed) {
            changed = false;

            auto is_sync = [](const node_t &node) -> bool {
                auto ss = std::dynamic_pointer_cast<StreamSync>(node);
                return bool(ss);
            };

            // search for two stream synchronize.
            // if they're the same stream, with no GPU operation between,
            // the second one won't do anything, so remove it.
            for (auto first = order.begin(); first != order.end(); ++first) {
                if (is_sync(*first)) {

                    // find next ss
                    auto second = first+1;
                    for (; second != order.end(); ++second) {
                        if (is_sync(*second)) {
                            break;
                        }
                    }

                    // no next stream sync
                    if (order.end() == second) {
                        break;
                    }


                    // two stream syncs, first and second
                    auto ss1 = std::dynamic_pointer_cast<StreamSync>(*first);
                    auto ss2 = std::dynamic_pointer_cast<StreamSync>(*second);
                    if (!ss1 || !ss2) THROW_RUNTIME("");

                    // synchronize the same stream
                    // if they don't, this might be a way of synchronizing two streams, so leave it in
                    if (ss1->stream() == ss2->stream()) {

                        // look for any GPU operations between them
                        bool gpuOpBetween = false;
                        for (auto it = first+1; it < second; ++it) {
                            if (auto gpu = std::dynamic_pointer_cast<StreamedOp>(*it)) {
                                gpuOpBetween = true;
                                break;
                            }
                        }

                        if (!gpuOpBetween) {
                            changed = true;
                            ++removed;
                            order.erase(second);
                            break; // out to while loop to search again
                        }
                    }
                }
            }
        }
    }

    // remove redundant event record / event syncs
    {
        bool changed = true;
        while(changed) {
            changed = false;

            // cudaEventRecord
            auto is_cer = [](const node_t &node) -> bool {
                auto cer = std::dynamic_pointer_cast<CudaEventRecord>(node);
                return bool(cer);
            };

            // search for two event records
            // if they're the same stream, with no GPU operation between,
            // the second one is the same as teh first one
            // remove it and it's associated event sync
            for (auto first = order.begin(); first != order.end(); ++first) {
                if (is_cer(*first)) {

                    // find next ss
                    auto second = first+1;
                    for (; second != order.end(); ++second) {
                        if (is_cer(*second)) {
                            break;
                        }
                    }

                    // no next cudaEventRecord
                    if (order.end() == second) {
                        break;
                    }


                    // two event records, first and second
                    auto cer1 = std::dynamic_pointer_cast<CudaEventRecord>(*first);
                    auto cer2 = std::dynamic_pointer_cast<CudaEventRecord>(*second);
                    if (!cer1 || !cer2) THROW_RUNTIME("");

                    // synchronize the same stream
                    if (cer1->stream() == cer2->stream()) {

                        // look for any GPU operations between them
                        bool gpuOpBetween = false;
                        for (auto it = first+1; it < second; ++it) {
                            if (auto gpu = std::dynamic_pointer_cast<StreamedOp>(*it)) {
                                gpuOpBetween = true;
                                break;
                            }
                        }

                        if (!gpuOpBetween) {
                            changed = true;
                            removed += 2;

                            // find the cudaStreamSync which uses that event and remove it
                            {
                                bool found = false;
                                for (auto it = second+1; it != order.end(); ++it) {
                                    auto ces = std::dynamic_pointer_cast<CudaEventSync>(*it);
                                    if (ces) {
                                        if (ces->event() == cer2->event()) {
                                            order.erase(it); // TODO: invalidates second?
                                            found = true;
                                            break;
                                        }
                                    }
                                }
                                if (!found) {
                                    THROW_RUNTIME("couldn't find CudaEventSync for unneeded CudaEventRecord");
                                }
                            }
                            order.erase(second); // remove the second event record
                            break; // out to while loop to search again
                        }
                    }
                }
            }
        }
    }


    return removed;
}

std::vector<Schedule> make_schedules(Graph<CpuNode> &g)
{
    std::vector<Schedule> currs; // input generation
    std::vector<Schedule> ret;

    {
        Schedule s;
        s.remaining.insert(g.start());
        currs.push_back(s);
    }

    while (!currs.empty())
    {
        // all possible schedules derived from the current schedules.
        // The current schedule, with all possible next legal operations
        std::vector<Schedule> nexts;

        std::cerr << "currs.size()=" << currs.size() << "\n";

        for (Schedule &curr : currs)
        {
            // if no more allowed operations, the schedule
            // must be complete
            if (curr.remaining.empty())
            {
                ret.push_back(curr);
            }
            else
            {
                // create schedules which are `curr` followed by each legal next operation
                for (std::shared_ptr<CpuNode> nextOp : curr.remaining)
                {

                    // check if nextOp's preds are all done
                    bool allDone = true;
                    for (std::shared_ptr<CpuNode> check : g.preds_[nextOp])
                    {
                        if (curr.order.end() == std::find(curr.order.begin(), curr.order.end(), check))
                        {
                            allDone = false;
                            break;
                        }
                    }

                    // if all preds are done, nextOp can be next
                    if (allDone)
                    {
                        Schedule next = curr;
                        next.remaining.erase(nextOp);
                        next.order.push_back(nextOp);
                        for (const std::shared_ptr<CpuNode> &succ : g.succs_[nextOp])
                        {
                            next.remaining.insert(succ);
                        }
                        nexts.push_back(next);
                    }
                }
            }
        }
        currs = std::move(nexts);
    }

    return ret;
};

/*

*/
std::vector<Schedule> make_schedules_random(Graph<CpuNode> &g, size_t n)
{
    typedef std::shared_ptr<CpuNode> node_t;

    // weight nodes by path to the end
    #warning multi-rank ordering
    std::map<node_t, int> pathsToEnd;
    {
        node_t end;
        // find end node and set to 1
        for (auto &node : g.succs_) {
            if (node.first->name() == "end") {
                end = node.first;
            }
        }
        pathsToEnd[end] = 1;

        // iteratively push up paths from bottom
        bool changed = true;
        while(changed) {
            changed = false;

            for (auto &kv : g.succs_) {
                if (end == kv.first) {
                    continue; // don't update end
                }
                auto it = pathsToEnd.insert(std::make_pair(kv.first, 0));
                int curVal = it.first->second;
                int newVal = 0;
                for (auto &succ : kv.second) {
                    auto it2 = pathsToEnd.insert(std::make_pair(succ, 0));
                    newVal += it2.first->second;
                }
                pathsToEnd[kv.first] = newVal;
                if (curVal != newVal) {
                    changed = true;
                }
            }

        }
    }


    // for (auto &kv : pathsToEnd) {
    //     std::cerr << kv.first->name() << ":" << kv.second << "\n";
    // }

    for (auto &kv : pathsToEnd) {
        if (0 == kv.second) {
            THROW_RUNTIME("operation " << kv.first->name() << " has no path to end");
        }
    }

    std::vector<Schedule> ret;

    for (size_t i = 0; i < n; ++i) {
        std::vector<node_t> frontier; // the allowable next nodes to execute
        frontier.push_back(g.start());

        Schedule sched;
        while (!frontier.empty()) {

            // select the next node randomly from the frontier of possible next nodes
            #warning random seed
            node_t selected;
            {
                // sum up all weights in frontier
                int totalWeight = 0;
                for (auto &node : frontier) {
                    totalWeight += pathsToEnd[node];
                }

                int sel = rand() % totalWeight;

                // count up weights until we pass the random number, use the node that
                // would have pushed us past the random number
                int runningTotal = 0;
                for (auto &node : frontier) {
                    runningTotal += pathsToEnd[node];
                    if (runningTotal >= sel) {
                        auto it = frontier.begin() + (rand() % frontier.size());
                        selected = *it;
                        frontier.erase(it);
                        break;
                    }
                }
            }
            if (!selected) THROW_RUNTIME("failed to select node");
            sched.order.push_back(selected);




            // std::cerr << "selected " << selected->name() << "\n";

            // all the selected node's successors who have all their preds visited and are not themselves visited to the frontier
            for (auto &succ : g.succs_.at(selected)) {

                // try next if succ already visited
                {
                    auto it = std::find(sched.order.begin(), sched.order.end(), succ);
                    if (it != sched.order.end()) {
                        // std::cerr << (*it)->name() << " already in schedule\n";
                        continue;
                    }
                }

                // if a successors's pred has not already been visited, skip it
                bool allPredsVisited = true;
                for (auto &succPred : g.preds_.at(succ)) {
                    auto it = std::find(sched.order.begin(), sched.order.end(), succPred);
                    if (it == sched.order.end()) {
                        // std::cerr << succ->name() << " has unvisited pred (" << succPred->name()<< ")\n";
                        allPredsVisited = false;
                        break;
                    }
                }
                if (!allPredsVisited) {
                    continue;
                }

                // std::cerr << "added successor " << succ->name() << " to frontier\n";
                frontier.push_back(succ);
            }
        }

        // for (auto &e : sched.order) {
        //     std::cerr << " " << e->name();
        // }
        // std::cerr << "\n\n";

        ret.push_back(sched);
    }
    return ret;
};

/* true if two schedules are identical under a stream bijection
*/
bool Schedule::predicate(const Schedule &a, const Schedule &b)
{

    // different sizes can't be equal
    if (a.order.size() != b.order.size())
    {
        return false;
    }

    /* for each otherwise equivalent node in a and b, a mapping between the streams used in node
       a and b
    */
    std::map<cudaStream_t, cudaStream_t> bij;

    // if a's stream matches b's under bijection, or new bijection entry,
    // return true. else return false.
    auto check_or_update_bijection = 
    [&](cudaStream_t sa, cudaStream_t sb) -> bool {
        if (bij.count(sa) && sb != bij[sa]) {
            return false;
        }
        if (bij.count(sb) && sa != bij[sb]) {
            return false;
        }
        bij[sa] = sb;
        bij[sb] = sa;
        return true;
    };



    for (size_t i = 0; i < a.order.size(); ++i)
    {
        auto ap = a.order[i];
        auto bp = b.order[i];

        // if the two nodes are not functionally equal
        if (!(ap->eq(bp))) {
            return false;
        }

        // two nodes could be equivalent, but the entire schedule must follow a consistent
        // mapping of streams.
        {
            auto aa = std::dynamic_pointer_cast<StreamedOp>(ap);
            auto bb = std::dynamic_pointer_cast<StreamedOp>(bp);
            if (aa && bb)
            {
                if (!check_or_update_bijection(aa->stream(), bb->stream())) return false;
            }
        }

        // StreamSync must follow stream mapping
        {
            auto aa = std::dynamic_pointer_cast<StreamSync>(ap);
            auto bb = std::dynamic_pointer_cast<StreamSync>(bp);
            if (aa && bb)
            {
                if (!check_or_update_bijection(aa->stream(), bb->stream())) return false;
            }
        }

        // StreamWait must follow stream mapping
        {
            auto aa = std::dynamic_pointer_cast<StreamWait>(ap);
            auto bb = std::dynamic_pointer_cast<StreamWait>(bp);
            if (aa && bb)
            {
                if (!check_or_update_bijection(aa->waiter(), bb->waiter())) return false;
                if (!check_or_update_bijection(aa->waitee(), bb->waitee())) return false;
            }
        }

    }

    return true;
#undef CHECK
}



bool Schedule::by_node_typeid(const Schedule &a, const Schedule &b) {

    return std::lexicographical_compare(
        a.order.begin(), a.order.end(),
        b.order.begin(), b.order.end(),
        [](const std::shared_ptr<Node> &i, const std::shared_ptr<Node> &j) {

            if (i->tag() < j->tag()) {
                return true;
            } else if (i->tag() > j->tag()) {
                return false;
            } else {
                {
                    auto si = std::dynamic_pointer_cast<StreamedOp>(i);
                    auto sj = std::dynamic_pointer_cast<StreamedOp>(j);
                    if (si && sj) {
                        return si->stream() < sj->stream();
                    }
                }
                {
                    auto si = std::dynamic_pointer_cast<StreamSync>(i);
                    auto sj = std::dynamic_pointer_cast<StreamSync>(j);
                    if (si && sj) {
                        return si->stream() < sj->stream();
                    }
                }
                {
                    auto si = std::dynamic_pointer_cast<StreamWait>(i);
                    auto sj = std::dynamic_pointer_cast<StreamWait>(j);
                    if (si && sj) {
                        if (si->waiter() < sj->waiter()) {
                            return true;
                        } else if (si->waiter() > sj->waiter()) {
                            return false;
                        } else {
                            return si->waitee() < sj->waitee();
                        }
                    }
                }
            }

            return false;
         }
    );
}

std::vector<BenchResult> Schedule::benchmark(std::vector<Schedule> &schedules, MPI_Comm comm, const BenchOpts &opts) {


    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // order to run schedules in each iteration
    std::vector<int> perm(schedules.size());
    std::iota(perm.begin(), perm.end(), 0);

    // each iteration's time for each schedule
    std::vector<std::vector<double>> times(schedules.size());

    // each iteration, do schedules in a random order
    for (size_t i = 0; i < opts.nIters; ++i) {
        if (0 == rank) {
            std::cerr << " " << i;
        }
        if (0 == rank) {
            std::random_shuffle(perm.begin(), perm.end());
        }
        MPI_Bcast(perm.data(), perm.size(), MPI_INT, 0, MPI_COMM_WORLD);
        for (int si : perm)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            double rstart = MPI_Wtime();
            schedules[si].run();
            double elapsed = MPI_Wtime() - rstart;
            times[si].push_back(elapsed);
        }
    }
    if (0 == rank) {
        std::cerr << std::endl;
    }

    // for each schedule
    for (size_t si = 0; si < times.size(); ++si)
    {
        // each iteration's time is the maximum observed across all ranks
        MPI_Allreduce(MPI_IN_PLACE, times[si].data(), times[si].size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    std::vector<BenchResult> ret;
    for (auto &st : times)
    {
        std::sort(st.begin(), st.end());
        BenchResult result;
        result.pct01  = st[st.size() * 01 / 100];
        result.pct10  = st[st.size() * 10 / 100];
        result.pct50  = st[st.size() * 50 / 100];
        result.pct90  = st[st.size() * 90 / 100];
        result.pct99  = st[st.size() * 99 / 100];
        result.stddev = stddev(st);
        ret.push_back(result);
    }
    return ret;
}