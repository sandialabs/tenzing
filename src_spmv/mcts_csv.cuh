/* use MCTS on a particular assignment of operations to streams

   yl is in stream1, other GPU operations in stream2
*/

#include "sched/cuda_runtime.h"
#include "sched/schedule.hpp"
#include "sched/graph.hpp"
#include "sched/numeric.hpp"
#include "sched/mcts.hpp"
#include "sched/benchmarker.hpp"

#include "ops_spmv.cuh"

#include "where.hpp"
#include "csr_mat.hpp"
#include "row_part_spmv.cuh"




typedef int Ordinal;
typedef float Scalar;

template <Where w>
using csr_type = CsrMat<w, Ordinal, Scalar>;

template <typename Strategy>
int mcts_csv(int argc, char **argv)
{



    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        STDERR("expected argument");
        exit(1);
    }
    CsvBenchmarker benchmarker(argv[1]);

    MPI_Barrier(MPI_COMM_WORLD);


    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    /* ensure streams are numerically ordered, so that later when ranks sort by stream,
       stream1 is the smallest on both ranks
    */
    if (stream1 > stream2) std::swap(stream1, stream2);



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

    std::shared_ptr<StreamedOp> scatter;
    {
        Scatter::Args args{
            .dst = spmv.x_send_buf().view(),
            .src = spmv.lx().view(),
            .idx = spmv.x_send_idx().view()};
        auto _scatter = std::make_shared<Scatter>(args);
        scatter = std::make_shared<StreamedOp>(_scatter, stream2);
    }

    std::shared_ptr<StreamedOp> yl, yr;
    {
        SpMV<Ordinal, Scalar>::Args rArgs, lArgs;
        rArgs.a = spmv.rA().view();
        rArgs.y = spmv.ly().view(); // FIXME: remote y?
        rArgs.x = spmv.rx().view();
        lArgs.a = spmv.lA().view();
        lArgs.y = spmv.ly().view();
        lArgs.x = spmv.lx().view();
        auto _yl = std::make_shared<SpMV<Ordinal, Scalar>>("yl", lArgs);
        auto _yr = std::make_shared<SpMV<Ordinal, Scalar>>("yr", rArgs);

        // yl and yr in different streams
        yl = std::make_shared<StreamedOp>(_yl, stream1);
        yr = std::make_shared<StreamedOp>(_yr, stream2);
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
    std::shared_ptr<StreamedOp> y;
    {
        VectorAdd::Args args;
        auto _y = std::make_shared<VectorAdd>("y", args);
        y = std::make_shared<StreamedOp>(_y, stream2);
    }
    std::shared_ptr<End> end = std::make_shared<End>();

    std::cerr << "create graph\n";
    Graph<Node> orig(start);

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

    std::vector<Graph<Node>> gpuGraphs;
    gpuGraphs.push_back(orig);

    if (0 == rank) {
        std::cerr << gpuGraphs.size() << " GpuNode graphs\n";
    }

    if (0 == rank) {
#if 1
        for (auto &graph : gpuGraphs) {
            graph.dump();
            std::cerr << "\n";
        }
#endif
#if 0
        gpuGraphs.begin()->dump();
        std::cerr << "\n";
        (--gpuGraphs.end())->dump();
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);

    
    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "insert sync...\n";
    std::vector<Graph<Node>> syncedGraphs;
    for (auto &graph : gpuGraphs) {
        auto next = insert_synchronization(graph);
        syncedGraphs.push_back(next);
    }

    if (0 == rank) {
        std::cerr << "created " << syncedGraphs.size() << " sync graphs:\n";
    }


    if (0 == rank) {
        syncedGraphs.begin()->dump();
        std::cerr << "\n";
    }



    if (0 == rank) {
        for (auto &graph : syncedGraphs) {
            graph.dump();
            std::cerr << "\n";
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "convert to cpu graphs...\n";
    std::vector<Graph<CpuNode>> cpuGraphs;
    for (auto &graph : syncedGraphs) {
        cpuGraphs.push_back(graph.nodes_cast<CpuNode>());
    }
    if (0 == rank) std::cerr << "converted " << cpuGraphs.size() << " graphs\n";


    mcts::Opts opts;
    opts.dumpTreePrefix = "spmv";
    opts.benchOpts.nIters = 50;


    STDERR("mcts...");
    opts.dumpTreeEvery = 100;
    opts.nIters = 200;
    mcts::Result result = mcts::mcts<Strategy>(cpuGraphs[0], benchmarker, MPI_COMM_WORLD, opts);

    
    for (const auto &simres : result.simResults) {
        std::cout << simres.benchResult.pct10 << ",";
        for (const auto &op : simres.path) {
            std::cout << op->name() << ",";
        }
        std::cout << "\n"; 
    }

    return 0;
    

}