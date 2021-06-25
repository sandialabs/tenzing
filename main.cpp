#include <iostream>



#include "fake_mpi.hpp"
#include "fake_cuda.hpp"
#include "ops_spmv.hpp"
#include "schedule.hpp"


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