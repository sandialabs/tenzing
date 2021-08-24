#include "sched/operation.hpp"

#include "ops_halo_exchange.hpp"

#include <mpi.h>

#include <vector>
#include <memory>

int main(int argc, char **argv) {

    typedef HaloExchange::StorageOrder StorageOrder;
    typedef HaloExchange::Args Args;

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int nQ; // quantities per gridpoint
    int nGhost; // ghost cell radius
    int nX, nY; // x and y extent of cells / rank
    size_t pitch; // pitch of allocated memory in bytes
    StorageOrder storageOrder = StorageOrder::QXY;
    typedef double Real;



    Args args;
    args.nQ = 3;
    args.nX = 128;
    args.nY = 128;
    args.pitch = 512;
    args.nGhost = 3;


    /* allocate width * height * depth
    */
    std::vector<Real> hostGrid;
    {
        size_t width, height, depth;
        switch(storageOrder) {
            case StorageOrder::QXY: {
                width = (nX + pitch - 1) / pitch * pitch;
                height = nX + 2 * nGhost;
                depth = nY + 2 * nGhost;
            }
        }

        hostGrid.resize(width * height * depth);
    }


    args.rankToCoord = [](int rank) -> Dim2<size_t> {
        Dim2<size_t> coord;
        #warning skeleton
        return coord;
    };
    args.coordToRank = [](const Dim2<size_t> &coord) -> int {
        int rank;
        #warning skeleton
        return rank;
    };


    std::cerr << "create nodes\n";
    std::shared_ptr<Start> start = std::make_shared<Start>();
    std::shared_ptr<HaloExchange> exchange = std::make_shared<HaloExchange>(args);
    std::shared_ptr<End> end = std::make_shared<End>();

    std::cerr << "create graph\n";
    Graph<Node> orig(start);
    orig.then(start, exchange);
    orig.then(exchange, end);

    std::cerr << "expand\n";
    exchange->expand_in(orig);


    MPI_Finalize();
    return 0;
}