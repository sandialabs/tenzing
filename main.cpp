#include <iostream>

// wrappers
#include "mpi.h"
#include "cuda_runtime.h"

#include "ops_spmv.cuh"
#include "schedule.hpp"

#include "where.hpp"
#include "csr_mat.hpp"
#include "row_part_spmv.cuh"

typedef int Ordinal;
typedef float Scalar;

template <Where w>
using csr_type = CsrMat<w, Ordinal, Scalar>;
using coo_type = CooMat<Ordinal, Scalar>;

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int m = 100;
    int n = m;
    int bw = m / size;
    int nnz = 200;

    csr_type<Where::host> A;
    // generate and distribute A
    if (0 == rank)
    {
        std::cerr << "generate matrix\n";
        A = random_band_matrix<Ordinal, Scalar>(m, bw, nnz);
    }

    RowPartSpmv<Ordinal, Scalar> spmv(A, 0, MPI_COMM_WORLD);

    std::cerr << "early exit\n";
    exit(1);

    Start *start = new Start();

    Scatter *scatter;
    {
        Scatter::Args args{
            .dst = spmv.x_send_buf().view(),
            .src = spmv.lx().view(),
            .idx = spmv.x_send_idx().view()};
        scatter = new Scatter(args, stream1);
    }

    SpMV<Ordinal, Scalar> *yl, *yr;
    {
        SpMV<Ordinal, Scalar>::Args rArgs{
            .a = spmv.rA().view(),
            .y = spmv.ly().view(), // FIXME: remote y?
            .x = spmv.rx().view()};
        SpMV<Ordinal, Scalar>::Args lArgs{
            .a = spmv.lA().view(),
            .y = spmv.ly().view(),
            .x = spmv.lx().view()};
        yl = new SpMV<Ordinal, Scalar>("yl", lArgs, stream2);
        yr = new SpMV<Ordinal, Scalar>("yr", rArgs, stream2);
    }

    PostSend *postSend;
    WaitSend *waitSend;
    {
        PostSend::Args args;
        for (auto &arg : spmv.send_params())
        {
            args.sends.push_back(IsendArgs{
                .buf = spmv.x_send_buf().data() + arg.displ,
                .count = arg.count,
                .datatype = MPI_FLOAT,
                .dest = arg.dst,
                .tag = 0,
                .comm = MPI_COMM_WORLD,
                .request = &arg.req});
        }
        postSend = new PostSend(args);
        waitSend = new WaitSend(args);
    }

    PostRecv *postRecv;
    WaitRecv *waitRecv;
    {
        PostRecv::Args args;
        for (auto &arg : spmv.recv_params())
        {
            args.recvs.push_back(IrecvArgs{
                .buf = spmv.rx().data() + arg.displ,
                .count = arg.count,
                .datatype = MPI_FLOAT,
                .source = arg.src,
                .tag = 0,
                .comm = MPI_COMM_WORLD,
                .request = &arg.req});
        }
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

    if (0 == rank) {
        std::cerr << schedules.size() << " schedules:\n";
        for (Schedule &s : schedules)
        {
            for (Operation *op : s.order)
            {
                std::cerr << op->name() << ", ";
            }
            std::cerr << "\n";
        }
    }


    for (Schedule &sched : schedules)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        sched.run();
    }

    MPI_Finalize();
}