#include "sched/operation.hpp"
#include "sched/schedule.hpp"

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

    cudaStream_t stream1, stream2;
    CUDA_RUNTIME(cudaStreamCreate(&stream1));
    CUDA_RUNTIME(cudaStreamCreate(&stream2));
    if (stream1 > stream2) std::swap(stream1, stream2);

    
    Args args;
    args.nQ = 3; // quantities per gridpoint
    args.nX = 128; // x and y extent of cells / rank
    args.nY = 128;
    args.pitch = 512; // pitch of allocated memory in bytes
    args.nGhost = 3; // ghost cell radius
    args.storageOrder = StorageOrder::QXY;


    /* allocate width * height * depth
    */
    std::vector<Real> hostGrid;
    {
        size_t width, height, depth;
        switch(args.storageOrder) {
            case StorageOrder::QXY: {
                width = (args.nX + args.pitch - 1) / args.pitch * args.pitch;
                height = args.nX + 2 * args.nGhost;
                depth = args.nY + 2 * args.nGhost;
            }
        }

        hostGrid.resize(width * height * depth);
        CUDA_RUNTIME(cudaMalloc(&args.grid, width * height * depth));
    }

    // rank dimensions
    Dim2<int64_t> rd(1,1);

    if (size != rd.x * rd.y) {
        THROW_RUNTIME("size " << size << " did not match rank dims\n");
    }

    args.rankToCoord = [rd](int _rank) -> Dim2<int64_t> {
        Dim2<int64_t> coord;
        coord.x = _rank % rd.x;
        coord.y = _rank / rd.x;
        return coord;
    };
    args.coordToRank = [size, rd](const Dim2<int64_t> &coord) -> int {

        Dim2<int64_t> wrapped(coord);

        // wrap out of bounds
        while(wrapped.x < 0) {
            wrapped.x += rd.x;
        }
        while(wrapped.y < 0) {
            wrapped.y += rd.y;
        }
        wrapped.x = wrapped.x % rd.x;
        wrapped.y = wrapped.y % rd.y;

        int _rank = wrapped.x + wrapped.y * rd.x;
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

    std::cerr << "expand\n";
    exchange->expand_in(orig);

    if (0 == rank) {
        std::cerr << "dump\n";
        orig.dump_graphviz("expanded.dot");
    }

    std::cerr << "assign streams\n";
    std::vector<Graph<Node>> gpuGraphs = use_streams2(orig, {stream1, stream2});

    if (0 == rank) {
        std::cerr << "dump\n";
        gpuGraphs[0].dump_graphviz("gpu_0.dot");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "insert sync...\n";
    std::vector<Graph<Node>> syncedGraphs;
    for (auto &graph : gpuGraphs) {
        auto next = insert_synchronization(graph);
        syncedGraphs.push_back(next);
    }

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

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "create orderings...\n";
    std::vector<Schedule> schedules;
    for (auto &graph : cpuGraphs) {
        auto ss = make_schedules_random(graph, 10);
        for (auto &s : ss) {
            schedules.push_back(s);
        }
    }
    std::cerr << "created " << schedules.size() << " schedules\n";

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "sort schedules...\n";
    std::sort(schedules.begin(), schedules.end(), Schedule::by_node_typeid);


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

    if (0 == rank) std::cerr << "testing schedules...\n";
    for (size_t i = 0; i < schedules.size(); ++i) {
    // for (size_t i = 9; i < 10; ++i) {
        if (0 == rank) std::cerr << " " << i;
        MPI_Barrier(MPI_COMM_WORLD);
        schedules[i].run();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (0 == rank) std::cerr << std::endl;
    if (0 == rank) std::cerr << "done" << std::endl;



    MPI_Finalize();
    return 0;
}