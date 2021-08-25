#include "sched/schedule.hpp"

#include "sched/macro_at.hpp"

#include <algorithm>
#include <typeinfo>
#include <typeindex>

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