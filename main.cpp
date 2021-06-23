#include <vector>
#include <set>
#include <iostream>
#include <string>

#ifndef CUDA_DEFINES
typedef int cudaStream_t;
typedef int cudaError_t;
cudaError_t cudaStreamCreate(cudaStream_t *stream) {
    static int num = 0;
    *stream=(++num); return 0;
}
#else 
#include <cuda_runtime.h>
#endif

#include "fake_mpi.hpp"

class Operation
{
public:
    virtual void run() {}

    std::set<Operation *> succs;
    std::set<Operation *> preds;

    // do op after this
    Operation *then(Operation *op)
    {
        succs.insert(op);
        op->preds.insert(this);
        return op;
    }

    virtual ~Operation(){};
    virtual std::string name() { return "<anon>"; }

    /*this operation replaces itself with the provided operation*/
    void replace_with(Operation *op)
    {
        // replace self in successors' preds
        for (Operation *succ : succs)
        {
            succ->preds.erase(this);
            succ->preds.insert(op);
        }

        // replace self in predecessors succs
        for (Operation *pred : preds)
        {
            pred->succs.erase(this);
            pred->succs.insert(op);
        }

        // replace op's preds and succs with mine
        op->succs = succs;
        op->preds = preds;

        succs.clear();
        preds.clear();
    }
};

class Start : public Operation
{
public:
    std::string name() override { return "Start"; }
};

class End : public Operation
{
public:
    std::string name() override { return "End"; }
};

class SpMV : public Operation
{
public:
    struct Args {
        float *y;
        int yN;
        float *x;
        int xN;
        int *rowPtr;
        int *colInd;
        float *colVal;
        int nRow;
    };

    std::string name_;
    Args args_;
    cudaStream_t stream_;
    SpMV(const std::string name, Args args, cudaStream_t stream) : name_(name), args_(args), stream_(stream) {}
    std::string name() override { return name_ + "(" + std::to_string(uintptr_t(stream_)) + ")"; }
};

/* y[i] += a[i]
*/
class VectorAdd : public Operation
{
public:
struct Args {
    float *y;
    float *a;
    int n;
};
    std::string name_;
    cudaStream_t stream_;
    Args args_;
    VectorAdd(const std::string name, Args args, cudaStream_t stream) : name_(name), args_(args), stream_(stream) {}
    std::string name() override { return name_ + "(" + std::to_string(uintptr_t(stream_)) + ")"; }
};

/* 
   dst[i] = src[idx[i]]
   dst[0..dstN]
   src[0..srcN]
   idx[0..srcN]
*/
class Scatter : public Operation
{
public:
    struct Args {
        float *dst;
        int dstN;
        float *src;
        int srcN;
        int *idx;
    };
    Args args_;
    cudaStream_t stream_;
    Scatter(Args args, cudaStream_t stream) : args_(args), stream_(stream) {}
    std::string name() override { return "Scatter(" + std::to_string(uintptr_t(stream_)) + ")"; }
};

class StreamSync : public Operation
{
public:
    cudaStream_t stream_;
    StreamSync(cudaStream_t stream) : stream_(stream) {}
    std::string name() override { return "StreamSync(" + std::to_string(uintptr_t(stream_)) + ")"; }
};

class PostRecv : public Operation
{
public:
    struct Args {
        std::vector<IrecvArgs> recvs; 
    };
    Args args_;
    PostRecv(Args args) : args_(args) {}
    std::string name() override { return "PostRecv"; }
};

class WaitRecv : public Operation
{
public:
    typedef PostRecv::Args Args;
    Args args_;
    WaitRecv(Args args) : args_(args) {}
    std::string name() override { return "WaitRecv"; }
};

class PostSend : public Operation
{
public:
    struct Args {
        std::vector<IsendArgs> sends; 
    };
    Args args_;
    PostSend(Args args) : args_(args) {}
    std::string name() override { return "PostSend"; }
};

class WaitSend : public Operation
{
public:
    typedef PostSend::Args Args;
    Args args_;
    WaitSend(Args args) : args_(args) {}
    std::string name() override { return "WaitSend"; }
};

class Schedule
{
public:
    std::set<Operation *> remaining; // possible next operations
    std::vector<Operation *> order;

    void run()
    {
        for (Operation *op : order)
        {
            op->run();
        }
    }
};

std::vector<Schedule> make_schedules(Operation *start)
{
    std::vector<Schedule> currs; // input generation
    std::vector<Schedule> ret;

    {
        Schedule s;
        s.remaining.insert(start);
        currs.push_back(s);
    }

    while (!currs.empty())
    {
        // all possible schedules derived from current schedules
        // followed by legal next operations
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
                for (Operation *nextOp : curr.remaining)
                {

                    // check if nextOp's preds are all done
                    bool allDone = true;
                    for (Operation *check : nextOp->preds)
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
                        for (Operation *succ : nextOp->succs)
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

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    Start *start = new Start();

    Scatter *scatter;
    {
        Scatter::Args args;
        scatter = new Scatter(args, stream1);
    }

    SpMV *yl, *yr;
    {
        SpMV::Args rArgs, lArgs;
        yl = new SpMV("yl", lArgs, stream2);
        yr = new SpMV("yr", rArgs, stream2);
    }

    PostSend *postSend;
    WaitSend *waitSend;
    {
        PostSend::Args args;
        postSend = new PostSend(args);
        waitSend = new WaitSend(args);
    }    
    
    PostRecv *postRecv;
    WaitRecv *waitRecv;
    {
        PostRecv::Args args;
        postRecv = new PostRecv(args);
        waitRecv = new WaitRecv(args);
    }
    VectorAdd *y;
    {
        VectorAdd::Args args;
        y = new VectorAdd("y", args, stream2);
    }
    StreamSync *waitScatter = new StreamSync(stream1);
    StreamSync *waitY = new StreamSync(stream2);
    End *end = new End();

    // immediately recv, local spmv, or scatter
    start->then(yl);
    start->then(postRecv);
    start->then(scatter)->then(waitScatter)->then(postSend);

    // remote matrix after recv
    waitRecv->then(yr)->then(end);

    // add after local and remote done, then end
    yl->then(y);
    yr->then(y);

    // end once add and send is done
    y->then(waitY)->then(end);
    waitSend->then(end);

    // initiate sends and recvs before waiting for either
    postSend->then(waitSend);
    postSend->then(waitRecv);
    postRecv->then(waitSend);
    postRecv->then(waitRecv);

    std::vector<Schedule> schedules = make_schedules(start);

    std::cerr << schedules.size() << " schedules:\n";
    for (Schedule &s : schedules)
    {
        for (Operation *op : s.order)
        {
            std::cerr << op->name() << ", ";
        }
        std::cerr << "\n";
    }

    for (Schedule &sched : schedules)
    {
        sched.run();
    }

    MPI_Finalize();
}