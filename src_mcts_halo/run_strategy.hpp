#include "sched/numeric.hpp"
#include "sched/operation.hpp"
#include "sched/schedule.hpp"
#include "sched/mcts.hpp"

#include "ops_halo_exchange.hpp"

#include <mpi.h>

#include <vector>
#include <memory>
#include <algorithm>

template <typename Strategy>
int doit(int argc, char **argv) {
    (void) argc;
    (void) argv;

    typedef HaloExchange::StorageOrder StorageOrder;
    typedef HaloExchange::Args Args;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    typedef double Real;

    // create two streams and ensure the numerically smaller one is first
    cudaStream_t stream1, stream2;
    CUDA_RUNTIME(cudaStreamCreate(&stream1));
    CUDA_RUNTIME(cudaStreamCreate(&stream2));
    if (stream1 > stream2) std::swap(stream1, stream2);

    
    Args args;
    args.nQ = 3; // quantities per gridpoint
    args.nX = 512; // x and y extent of cells / rank
    args.nY = 512;
    args.nZ = 512;
    args.pitch = 128; // pitch of allocated memory in bytes
    args.nGhost = 3; // ghost cell radius
    args.storageOrder = StorageOrder::XYZQ;


    /* allocate width * height * depth
    */
    {
        size_t pitch, d2, d3, d4;
        switch(args.storageOrder) {
            case StorageOrder::QXYZ: {
                pitch = (sizeof(double) * args.nQ + args.pitch - 1) / args.pitch * args.pitch;
                d2 = args.nX + 2 * args.nGhost;
                d3 = args.nY + 2 * args.nGhost;
                d4 = args.nZ + 2 * args.nGhost;
                break;
            }
            case StorageOrder::XYZQ: {
                pitch = round_up(sizeof(double) * (args.nX + 2 * args.nGhost), args.pitch);
                d2 = args.nY + 2 * args.nGhost;
                d3 = args.nZ + 2 * args.nGhost;
                d4 = args.nQ;
                break;
            }
            default:
            THROW_RUNTIME("unhandled storage order");
        }

        std::cerr << "alloc p=" << pitch << " d2=" << d2 << " d3=" << d3 << " d4=" << d4 
                  << " (" << pitch * d2 * d3 * d4 / 1024.0 / 1024.0 << "MiB)\n";
        CUDA_RUNTIME(cudaMalloc(&args.grid, pitch * d2 * d3 * d4));
    }

    // rank dimensions
    Dim3<int64_t> rd(1,1,1);

    {
        for (const auto &pf : prime_factors(size)) {
            if (rd.x < rd.y && rd.x < rd.z) {
                rd.x *= pf;
            } else if (rd.y < rd.z) {
                rd.y *= pf;
            } else {
                rd.z *= pf;
            }
        }
        if (0 == rank) std::cerr << "rank grid: " << rd << "\n";

    }

    if (size != rd.x * rd.y * rd.z) {
        THROW_RUNTIME("size " << size << " did not match rank dims\n");
    }

    args.rankToCoord = [rd](int _rank) -> Dim3<int64_t> {
        Dim3<int64_t> coord;
        coord.x = _rank % rd.x;
        _rank /= rd.x;
        coord.y = _rank % rd.y;
        _rank /= rd.y;
        coord.z = _rank % rd.z;
        return coord;
    };
    args.coordToRank = [size, rd](const Dim3<int64_t> &coord) -> int {

        Dim3<int64_t> wrapped(coord);

        // wrap out of bounds
        while(wrapped.x < 0) {
            wrapped.x += rd.x;
        }
        while(wrapped.y < 0) {
            wrapped.y += rd.y;
        }
        while(wrapped.z < 0) {
            wrapped.z += rd.z;
        }
        wrapped.x = wrapped.x % rd.x;
        wrapped.y = wrapped.y % rd.y;
        wrapped.z = wrapped.z % rd.z;

        int _rank = wrapped.x + wrapped.y * rd.x + wrapped.z * rd.x * rd.y;
        if (_rank >= size || _rank < 0) {
            THROW_RUNTIME("invalid computed rank " << _rank);
        }
        return _rank;
    };


    std::cerr << "create nodes\n";
    std::shared_ptr<Start> start = std::make_shared<Start>();
    std::shared_ptr<HaloExchange> exchange = std::make_shared<HaloExchange>(args);
    std::shared_ptr<End> end = std::make_shared<End>();

    std::cerr << "create graph\n";
    Graph<Node> orig(start);
    orig.then(start, exchange);
    orig.then(exchange, end);

    if (0 == rank) {
        orig.dump_graphviz("orig.dot");
    }

    std::cerr << "expand and assign streams\n";
    // exchange->expand_in(orig);
    // x in one stream, y/z in another
    exchange->expand_3d_streams(orig, stream1, stream2, stream2);

    if (0 == rank) {
        std::cerr << "dump\n";
        orig.dump_graphviz("expanded.dot");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "insert sync...\n";
    std::vector<Graph<Node>> syncedGraphs;
    syncedGraphs.push_back(insert_synchronization(orig));

    if (0 == rank) {
        std::cerr << "dump\n";
        syncedGraphs[0].dump_graphviz("sync_0.dot");
    }


    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "convert to cpu graphs...\n";
    std::vector<Graph<CpuNode>> cpuGraphs;
    for (auto &graph : syncedGraphs) {
        cpuGraphs.push_back(graph.nodes_cast<CpuNode>());
    }
    if (0 == rank) std::cerr << "converted " << cpuGraphs.size() << " graphs\n";

    mcts::Opts opts;
    opts.dumpTreePrefix = "halo";
    opts.benchOpts.nIters = 60;

    STDERR("mcts (warmup)");
    {
        opts.nIters = 40;
        mcts::mcts<Strategy>(cpuGraphs[0], MPI_COMM_WORLD, opts);        
    }

    STDERR("mcts...");
    opts.dumpTreeEvery = 500;
    opts.nIters = 2000;
    mcts::Result result = mcts::mcts<Strategy>(cpuGraphs[0], MPI_COMM_WORLD, opts);

    
    for (const auto &simres : result.simResults) {
        std::cout << simres.benchResult.pct10 << ",";
        for (const auto &op : simres.path) {
            std::cout << op->name() << ",";
        }
        std::cout << "\n"; 
    }

    return 0;
}