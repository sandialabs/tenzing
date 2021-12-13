/* use MCTS for both order and streams
*/

#include "sched/cuda_runtime.h"
#include "sched/schedule.hpp"
#include "sched/graph.hpp"
#include "sched/numeric.hpp"
#include "sched/mcts.hpp"
#include "sched/benchmarker.hpp"
#include "sched/platform.hpp"


#include "ops_spmv.cuh"

#include "where.hpp"
#include "csr_mat.hpp"
#include "row_part_spmv.cuh"


typedef int Ordinal;
typedef float Scalar;

template <Where w>
using csr_type = CsrMat<w, Ordinal, Scalar>;

template <typename Strategy>
int platform_mcts(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    EmpiricalBenchmarker benchmarker;

    MPI_Barrier(MPI_COMM_WORLD);

    /* interesting parameters:
       vortex: 1.5m rows, 15mnnz, bandwidth = 15m/16 4 nodes, 4 ranks per node
       may be even more interesting with 12 nodes, 4 ranks per node
    */

    int m = 150000;
    int bw = m / size;
    int nnz = m * 10;

    csr_type<Where::host> A;

    // generate and distribute A
    if (0 == rank)
    {
        std::cerr << "generate matrix\n";
        A = random_band_matrix<Ordinal, Scalar>(m, bw, nnz);
    }
    

    RowPartSpmv<Ordinal, Scalar> spmv(A, 0, MPI_COMM_WORLD);

    std::shared_ptr<Start> start = std::make_shared<Start>();

    std::shared_ptr<Scatter> scatter;
    {
        Scatter::Args args{
            .dst = spmv.x_send_buf().view(),
            .src = spmv.lx().view(),
            .idx = spmv.x_send_idx().view()};
        scatter = std::make_shared<Scatter>(args);
    }

    std::shared_ptr<SpMV<Ordinal, Scalar>> yl, yr;
    {
        SpMV<Ordinal, Scalar>::Args rArgs, lArgs;
        rArgs.a = spmv.rA().view();
        rArgs.y = spmv.ly().view(); // FIXME: remote y?
        rArgs.x = spmv.rx().view();
        lArgs.a = spmv.lA().view();
        lArgs.y = spmv.ly().view();
        lArgs.x = spmv.lx().view();
        yl = std::make_shared<SpMV<Ordinal, Scalar>>("yl", lArgs);
        yr = std::make_shared<SpMV<Ordinal, Scalar>>("yr", rArgs);
    }

    std::shared_ptr<PostSend> postSend;
    std::shared_ptr<WaitSend> waitSend;
    {
        PostSend::Args args;
        for (auto &arg : spmv.send_params())
        {
            if (arg.displ + arg.count > spmv.x_send_buf().size()) throw std::logic_error(AT);
            if (!spmv.x_send_buf().data()) throw std::logic_error(AT);
            args.sends.push_back(Isend::Args{
                .buf = spmv.x_send_buf().data() + arg.displ,
                .count = arg.count,
                .datatype = MPI_FLOAT,
                .dest = arg.dst,
                .tag = 0,
                .comm = MPI_COMM_WORLD,
                .request = &arg.req});
        }
        postSend = std::make_shared<PostSend>(args);
        waitSend = std::make_shared<WaitSend>(args);
    }

    std::shared_ptr<PostRecv> postRecv;
    std::shared_ptr<WaitRecv> waitRecv;
    {
        PostRecv::Args args;
        for (auto &arg : spmv.recv_params())
        {
            if (arg.displ + arg.count > spmv.rx().size()) throw std::logic_error(AT);
            if (!spmv.rx().data()) throw std::logic_error(AT);
            args.recvs.push_back(Irecv::Args{
                .buf = spmv.rx().data() + arg.displ,
                .count = arg.count,
                .datatype = MPI_FLOAT,
                .source = arg.src,
                .tag = 0,
                .comm = MPI_COMM_WORLD,
                .request = &arg.req});
        }
        postRecv = std::make_shared<PostRecv>(args);
        waitRecv = std::make_shared<WaitRecv>(args);
    }
    std::shared_ptr<VectorAdd> y;
    {
        VectorAdd::Args args;
        y = std::make_shared<VectorAdd>("y", args);
    }
    std::shared_ptr<End> end = std::make_shared<End>();

    std::cerr << "create graph\n";
    Graph<OpBase> orig(start);

    // immediately recv, local spmv, or scatter
    orig.then(start, yl);
    orig.then(start, postRecv);
    orig.then(orig.then(start, scatter), postSend);

    // remote matrix after recv
    orig.then(waitRecv, yr);

    // add after local and remote done, then end
    orig.then(yl, y);
    orig.then(yr, y);

    // end once add and send is done
    orig.then(y, end);
    orig.then(waitSend, end);

    // initiate sends and recvs before waiting for either
    orig.then(postSend, waitSend);
    orig.then(postSend, waitRecv);
    orig.then(postRecv, waitSend);
    orig.then(postRecv, waitRecv);

    orig.dump();
    MPI_Barrier(MPI_COMM_WORLD);


    if ( 0 == rank ) {
        std::cerr << "create platform";
    }
    Platform platform = Platform::make_n_streams(2, MPI_COMM_WORLD);

    STDERR("mcts...");
    mcts::Opts opts;
    opts.nIters = 300;
    opts.dumpTreePrefix = "spmv";
    opts.dumpTreeEvery = 100;
    opts.benchOpts.nIters = 50;
    mcts::Result result = mcts::mcts<Strategy>(orig, platform, benchmarker, opts);


    for (const auto &simres : result.simResults) {
        std::cout << simres.benchResult.pct10;
        for (const auto &op : simres.path) {
            std::cout << "|" << op->json();
        }
        std::cout << "\n"; 
    }

    return 0;
}