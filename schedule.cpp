#include "schedule.hpp"

#include "at.hpp"

#include <algorithm>

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
                        for (std::shared_ptr<CpuNode> succ : g.succs_[nextOp])
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
    std::map<cudaStream_t, cudaStream_t> streams;
    for (size_t i = 0; i < a.order.size(); ++i)
    {
        auto ap = a.order[i];
        auto bp = b.order[i];

        // if the two nodes are not functionally equal
        if (!(ap->equal(bp))) {
            return false;
        }

        // two nodes could be equivalent, but the entire schedule must follow a consistent
        // mapping of streams.
        {
            auto aa = std::dynamic_pointer_cast<StreamedOp>(ap);
            auto bb = std::dynamic_pointer_cast<StreamedOp>(bp);
            if (aa && bb)
            {
                // if we've seen a;'s stream before, we know which stream b should be using
                if (streams.count(aa->stream()))
                {
                    if (bb->stream() != streams[aa->stream()])
                    {
                        return false;
                    }
                }
                else // otherwise, record which stream in b corresponds to the stream in a
                {
                    streams[aa->stream()] = bb->stream();
                }
            }
        }

        // StreamSync must follow stream mapping
        {
            auto aa = std::dynamic_pointer_cast<StreamSync>(ap);
            auto bb = std::dynamic_pointer_cast<StreamSync>(bp);
            if (aa && bb)
            {
                if (streams.count(aa->stream()))
                {
                    if (bb->stream() != streams[aa->stream()])
                    {
                        return false;
                    }
                }
                else
                {
                    streams[aa->stream()] = bb->stream();
                }
            }
        }

        // StreamWait must follow stream mapping
        {
            auto aa = std::dynamic_pointer_cast<StreamWait>(ap);
            auto bb = std::dynamic_pointer_cast<StreamWait>(bp);
            if (aa && bb)
            {
                if (streams.count(aa->waiter()))
                {
                    if (bb->waiter() != streams[aa->waiter()])
                    {
                        return false;
                    }
                }
                else
                {
                    streams[aa->waiter()] = bb->waiter();
                }


                if (streams.count(aa->waitee()))
                {
                    if (bb->waitee() != streams[aa->waitee()])
                    {
                        return false;
                    }
                }
                else
                {
                    streams[aa->waitee()] = bb->waitee();
                }

            }
        }

    }

    return true;
#undef CHECK
}