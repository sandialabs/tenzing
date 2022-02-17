/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "sched/schedule.hpp"

#include "sched/macro_at.hpp"
#include "sched/numeric.hpp"
#include "sched/ops_cuda.hpp"
#include "sched/randomness.hpp"

#include <algorithm>
#include <typeinfo>
#include <typeindex>
#include <numeric>


int Schedule::remove_redundant_syncs(std::vector<std::shared_ptr<BoundOp>> &order) {
    using op_t = std::shared_ptr<BoundOp>;

    int removed = 0;

    auto is_css = [](const op_t &op) -> bool {
        return bool(std::dynamic_pointer_cast<StreamSync>(op));
    };
    auto is_cer = [](const op_t &op) -> bool {
        return bool(std::dynamic_pointer_cast<CudaEventRecord>(op));
    };


    auto find = [&](const op_t &op) -> std::vector<std::shared_ptr<BoundOp>>::iterator {
        auto it = order.begin();
        for (; it < order.end(); ++it) {
            if (*it == op) return it;
        }
        return it;
    };

    // true if two CER represent the same point in the same stream
    auto same_stream_state = [&](
        const std::shared_ptr<CudaEventRecord> &a,
        const std::shared_ptr<CudaEventRecord> &b) -> bool {

        if (a->stream() != b->stream()) {
            return false;
        }

        auto ai = find(a);
        auto bi = find(b);
        for (auto it = ai; it < bi; ++it) {
            if (auto gpu = std::dynamic_pointer_cast<BoundGpuOp>(*it)) {
                return false;
            }
        }
        return true;
    };

    
    // restart after any successful change
    bool changed = true;
enter_changed_loop:
    while (changed) {
        changed = false;


        // remove any CER that is never CSWE or CES
        for (auto first = order.begin(); first < order.end(); ++first) {
            if (auto cer = std::dynamic_pointer_cast<CudaEventRecord>(*first)) {

                bool erase = true; // assume this event should be erased

                // cancel erase if a later sync uses the event
                for (auto second = first+1; second < order.end(); ++second) {
                    if (auto ces = std::dynamic_pointer_cast<CudaEventSync>(*second)) {
                        if (cer->event() == ces->event()) {
                            erase = false;
                        }
                    }
                    if (auto cswe = std::dynamic_pointer_cast<CudaStreamWaitEvent>(*second)) {
                        if (cer->event() == cswe->event()) {
                            erase = false;
                        }
                    }
                }

                if (erase) {
                    ++removed;
                    order.erase(first);
                    changed = true;
                    goto enter_changed_loop;
                }
            }
        }

        // remove any CSWE where there is no GPU operation following in the stream
        // CER is cleaned up separately
        for (auto cswei = order.begin(); cswei < order.end(); ++cswei) {
            if (auto cswe = std::dynamic_pointer_cast<CudaStreamWaitEvent>(*cswei)) {
                // search for following GPU op in synchronized stream
                bool found = false;
                for (auto second = cswei+1; second < order.end(); ++second) {
                    if (auto bgo = std::dynamic_pointer_cast<BoundGpuOp>(*second)) {
                        if (bgo->stream() == cswe->stream()) {
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) {
                    order.erase(cswei);
                    changed = true;
                    ++removed;
                    goto enter_changed_loop;
                }
            }
        }

        // remove the first of any two CSS in the same stream if they represent the same state
        // remove the first to sync as late as possible
        for (auto first = order.begin(); first != order.end(); ++first) {
            if (is_css(*first)) {

                // find next ss
                auto second = first+1;
                for (; second != order.end(); ++second) {
                    if (is_css(*second)) {
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
                        if (auto gpu = std::dynamic_pointer_cast<BoundGpuOp>(*it)) {
                            gpuOpBetween = true;
                            break;
                        }
                    }

                    if (!gpuOpBetween) {
                        changed = true;
                        ++removed;
                        order.erase(first);
                        goto enter_changed_loop;
                    }
                }
            }
        }






        /* Search for two CERs that represent the same point in a stream
        remove the first one, and its corresponding CES to delay the sync as late as possible
        */
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

                // represent the same point in the execution
                if (same_stream_state(cer1, cer2)) {

                    // find the cudaEventSync which uses each event
                    auto ces1 = order.end();
                    auto ces2 = order.end();

                    // start search at first since sync 1 may come before record2
                    for (auto needle = first+1; needle != order.end(); ++needle) {
                        auto ces = std::dynamic_pointer_cast<CudaEventSync>(*needle);
                        if (ces) {
                            // found CER2's sync
                            if (ces->event() == cer1->event()) {
                                ces1 = needle;
                            } else if (ces->event() == cer2->event()) {
                                ces2 = needle;
                            }
                        }
                    }
                    if (ces1 == order.end() || ces2 == order.end()) {
                        THROW_RUNTIME("couldn't find cudaEventSync for cudaEventRecord");
                    }

                    // remove the event that's synced first
                    if (ces2 < ces1) {
                        order.erase(ces1);
                        order.erase(first);
                        changed = true;
                        removed += 2;
                        goto enter_changed_loop;
                    } else if (ces1 < ces2) {
                        order.erase(ces2);
                        order.erase(second);
                        changed = true;
                        removed += 2;
                        goto enter_changed_loop;
                    }
                }
            }
        }

        /* search for two event records (1, then 2) in the same stream

            if the first event is synced after the second is synced,
            it is guaranteed to have happened at the second sync
            and the first record/sync is not needed

            FIXME: This could be extended to CudaStreamWaitEvent, so long as the
            two streams that are waiting are the same, just like how
            CudaEventSyncs both sync the CPU
        */
        for (auto first = order.begin(); first != order.end(); ++first) {
            if (is_cer(*first)) {

                // find next cer
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

                // record the same stream
                if (cer1->stream() == cer2->stream()) {


                    // find ces1 and ces2
                    auto ces1 = order.end();
                    auto ces2 = order.end();

                    // start search at first cer since ces1 may come before cer2
                    for (auto needle = first+1; needle != order.end(); ++needle) {
                        auto ces = std::dynamic_pointer_cast<CudaEventSync>(*needle);
                        if (ces) {
                            if (ces->event() == cer1->event()) {
                                ces1 = needle;
                            } else if (ces->event() == cer2->event()) {
                                ces2 = needle;
                            }
                        }
                    }

                    // there may not be a CudaEventSync (e.g., CudaStreamWaitEvent instead)
                    // search for the next part of CER
                    if (order.end() == ces1 || order.end() == ces2) {
                        continue;
                    }

                    // sync for event 2 is first, remove first event & sync
                    if (ces2 < ces1) {
                        // remove last one first to not invalidate iterator
                        STDERR("remove " << (*ces1)->desc());
                        order.erase(ces1);
                        STDERR("remove " << (*first)->desc());
                        order.erase(first);
                        changed = true;
                        removed += 2;
                        goto enter_changed_loop;
                    }
                }
            }
        }
    } // changed loop



    {
        std::string s;
        for (const auto &op : order) {
            s += op->desc();
            s += ", ";
        }
        STDERR("remove_redundant_syncs result is: " << s);
    }

    return removed;
}

int Schedule::remove_redundant_syncs() {
    return remove_redundant_syncs(order);
}

std::vector<Schedule> make_schedules(Graph<BoundOp> &g)
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
                for (std::shared_ptr<BoundOp> nextOp : curr.remaining)
                {

                    // check if nextOp's preds are all done
                    bool allDone = true;
                    for (std::shared_ptr<BoundOp> check : g.preds_[nextOp])
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
                        for (const std::shared_ptr<BoundOp> &succ : g.succs_[nextOp])
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
std::vector<Schedule> make_schedules_random(Graph<BoundOp> &g, size_t n)
{
    typedef std::shared_ptr<BoundOp> op_t;

    // weight nodes by path to the end
    #warning multi-rank ordering
    std::map<op_t, int> pathsToEnd;
    {
        op_t end;
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
        std::vector<op_t> frontier; // the allowable next nodes to execute
        frontier.push_back(g.start());

        Schedule sched;
        while (!frontier.empty()) {

            // select the next node randomly from the frontier of possible next nodes
            #warning random seed
            op_t selected;
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
    std::map<Stream, Stream> bij;

    // if a's stream matches b's under bijection, or new bijection entry,
    // return true. else return false.
    auto check_or_update_bijection = 
    [&](Stream sa, Stream sb) -> bool {
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
            auto aa = std::dynamic_pointer_cast<BoundGpuOp>(ap);
            auto bb = std::dynamic_pointer_cast<BoundGpuOp>(bp);
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



bool Schedule::by_op_typeid(const Schedule &a, const Schedule &b) {

    return std::lexicographical_compare(
        a.order.begin(), a.order.end(),
        b.order.begin(), b.order.end(),
        [](const std::shared_ptr<OpBase> &i, const std::shared_ptr<OpBase> &j) {

            if (i->tag() < j->tag()) {
                return true;
            } else if (i->tag() > j->tag()) {
                return false;
            } else {
                {
                    auto si = std::dynamic_pointer_cast<BoundGpuOp>(i);
                    auto sj = std::dynamic_pointer_cast<BoundGpuOp>(j);
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



