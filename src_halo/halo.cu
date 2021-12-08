#include "sched/numeric.hpp"
#include "sched/operation.hpp"
#include "sched/schedule.hpp"
#include "sched/benchmarker.hpp"

#include "ops_halo_exchange.hpp"

#include <mpi.h>

#include <vector>
#include <memory>
#include <algorithm>

int main(int argc, char **argv) {

    typedef HaloExchange::StorageOrder StorageOrder;
    typedef HaloExchange::Args Args;

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    typedef double Real;

    Platform plat = Platform::make_n_streams(2, MPI_COMM_WORLD);
    
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
    Graph<OpBase> orig(start);
    orig.then(start, exchange);
    orig.then(exchange, end);

    if (0 == rank) {
        orig.dump_graphviz("orig.dot");
    }

#if 0
    std::cerr << "expand\n";
    exchange->expand_in(orig);

    if (0 == rank) {
        std::cerr << "dump\n";
        orig.dump_graphviz("expanded.dot");
    }

    std::cerr << "assign streams\n";
    std::vector<Graph<OpBase>> gpuGraphs = use_streams2(orig, {stream1, stream2});

    if (0 == rank) {
        std::cerr << "dump\n";
        gpuGraphs[0].dump_graphviz("gpu_0.dot");
    }


    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "insert sync...\n";
    std::vector<Graph<OpBase>> syncedGraphs;
    for (auto &graph : gpuGraphs) {
        auto next = insert_synchronization(graph);
        syncedGraphs.push_back(next);
    }

    if (0 == rank) {
        std::cerr << "dump\n";
        syncedGraphs[0].dump_graphviz("sync_0.dot");
        syncedGraphs[5].dump_graphviz("sync_5.dot");
    }
#else
    std::cerr << "expand and assign streams\n";
    exchange->expand_3d_streams(orig, stream1, stream2, stream2);

    if (0 == rank) {
        std::cerr << "dump\n";
        orig.dump_graphviz("expanded.dot");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "insert sync...\n";
    std::vector<Graph<OpBase>> syncedGraphs;
    syncedGraphs.push_back(insert_synchronization(orig));

    if (0 == rank) {
        std::cerr << "dump\n";
        syncedGraphs[0].dump_graphviz("sync_0.dot");
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "convert to cpu graphs...\n";
    std::vector<Graph<CpuOp>> cpuGraphs;
    for (auto &graph : syncedGraphs) {
        cpuGraphs.push_back(graph.nodes_cast<CpuOp>());
    }
    if (0 == rank) std::cerr << "converted " << cpuGraphs.size() << " graphs\n";

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "create orderings...\n";
    std::vector<Schedule> schedules;
    for (auto &graph : cpuGraphs) {
        auto ss = make_schedules_random(graph, 1000);
        for (auto &s : ss) {
            schedules.push_back(s);
        }
        std::cerr << ".";
    }
    std::cerr << "\n";
    std::cerr << "created " << schedules.size() << " schedules\n";

#if 0
    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t si = 10000; si < 10010; ++si) {
        for (auto &op : schedules[si].order) {
            if (0 == rank) std::cerr << "," << op->name();
        }
        if (0 == rank) std::cerr << "\n";
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "remove redundant syncs schedules...\n";
    for (auto &sched : schedules) {
        int count = sched.remove_redundant_syncs();
        if (0 == rank) std::cerr << count << " ";
    }
    if (0 == rank) std::cerr << "\n";

#if 0
    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t si = 10000; si < 10010; ++si) {
        for (auto &op : schedules[si].order) {
            if (0 == rank) std::cerr << "," << op->name();
        }
        if (0 == rank) std::cerr << "\n";
    }
#endif


    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "sort schedules...\n";
    std::sort(schedules.begin(), schedules.end(), Schedule::by_op_typeid);


    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "eliminate equivalent schedules...\n";
    {
        int count = 0;
        size_t total = schedules.size() * (schedules.size() - 1);
        int next = 99;
        for (size_t i = 0; i < schedules.size(); ++i) {
            for (size_t j = i+1; j < schedules.size(); ++j) {
                if (Schedule::predicate(schedules[i], schedules[j])) {
                    schedules.erase(schedules.begin() + j);
                    count += 1;
                    --j; // since we need to check the schedule that is now in j
                }
                size_t left = (schedules.size() - i) * (schedules.size() - i - 1);
                if (left < next * total / 100) {
                    if (0 == rank) std::cerr << next << "% (~" << (schedules.size()-i) * (schedules.size() - i - 1) << " comparisons left...)\n";
                    next = left * 100 / total;
                }
            }
        }
        std::cerr << "found " << count << " duplicate schedules\n";
        std::cerr << "found " << schedules.size() << " unique schedules\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);





    if (0 == rank) std::cerr << "testing schedules...\n";
    for (size_t i = 0; i < schedules.size(); ++i) {
    // for (size_t i = 53; i < 54; ++i) {
        if (0 == rank) std::cerr << " " << i;
        MPI_Barrier(MPI_COMM_WORLD);
        schedules[i].run();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (0 == rank) std::cerr << std::endl;
    if (0 == rank) std::cerr << "done" << std::endl;


    if (0 == rank) std::cerr << "benching schedules...\n";
    BenchOpts opts;
    opts.nIters = 100;
    EmpiricalBenchmarker benchmarker;
    auto benchResults = benchmarker.benchmark(schedules, MPI_COMM_WORLD, opts);
    if (0 == rank) std::cerr << "done" << std::endl;

    if (0 == rank)
    {
        std::cout << "1pctl,10pctl,50pctl,90pct,99pct,stddev,order\n";
        for (size_t si = 0; si < benchResults.size(); ++si) {
            auto &result = benchResults[si];
            std::cout 
                   << result.pct01 
            << "," << result.pct10 
            << "," << result.pct50 
            << "," << result.pct90 
            << "," << result.pct99 
            << "," << result.stddev;

            for (auto &op : schedules[si].order) {
                std::cout << "," << op->name();
            }

            std::cout << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}