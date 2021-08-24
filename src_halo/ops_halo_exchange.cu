#include "ops_halo_exchange.hpp"

#include "sched/ops_mpi.hpp"

#include "cuda_memory.hpp"

void HaloExchange::expand_in(Graph<Node> &g) {
    // find preds and successors of this graph
    Graph<Node>::NodeSet &preds = g.preds(this);
    Graph<Node>::NodeSet &succs = g.succs(this);

    // new nodes created to replace this node
    std::vector<std::shared_ptr<Isend>> sends;
    std::vector<std::shared_ptr<Irecv>> recvs;
    std::vector<std::shared_ptr<Wait>> waitSends;
    std::vector<std::shared_ptr<Wait>> waitRecvs;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    #warning fixme all these operations need unique names for sorting

    const Dim2<size_t> myCoord = args_.rankToCoord(rank);

    // create pack and send for each direction
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (0 != dx && 0 != dy) {

                Dim2<size_t> inbufExt(args_.nX + 2 * args_.nGhost, args_.nY + 2 * args_.nGhost);

                Dim2<size_t> inbufOff(args_.nGhost, args_.nGhost);
                if (1 == dx) {
                    inbufOff.x += args_.nX;
                }
                if (1 == dy) {
                    inbufOff.y += args_.nY;
                }

                Dim2<size_t> packExt(args_.nX, args_.nY);
                if (0 != dx) {
                    packExt.x = args_.nGhost;
                }
                if (0 != dy) {
                    packExt.y = args_.nGhost;
                }

                // create pack
                Pack::Args packArgs;
                packArgs.inbufOff = inbufOff;
                packArgs.packExt = packExt;
                packArgs.inbufExt = inbufExt;
                packArgs.pitch = args_.pitch;
                packArgs.nQ = args_.nQ;
                packArgs.storageOrder = args_.storageOrder;
                packArgs.inbuf = args_.grid;
                auto pack = std::make_shared<Pack>(packArgs);

                const Dim2<size_t> dstCoord = myCoord + Dim2<size_t>(dx, dy);

                // create Isend
                OwningIsend::Args sendArgs;
                sendArgs.buf = pack->outbuf();
                sendArgs.count = args_.nQ * args_.nX * args_.nY;
                sendArgs.datatype = MPI_DOUBLE;
                sendArgs.dest = args_.coordToRank(dstCoord);
                sendArgs.tag = 0;
                sendArgs.comm = MPI_COMM_WORLD;
                sendArgs.request = nullptr; // will be set to owned req
                auto send = std::make_shared<OwningIsend>(sendArgs);
                sends.push_back(send);

                
                // connect preds -> pack
                for (auto &pred : preds) {
                    g.then(pred, pack);
                }
                
                // connect pack -> Isend
                g.then(pack, send);

                // connect send -> succs
                for (auto &succ : succs) {
                    g.then(pack, succ);
                }
                
                // create a wait in waitSends
                {
                    Wait::Args args;
                    args.request = sendArgs.request;
                    args.status = MPI_STATUS_IGNORE;
                    waitSends.push_back(std::make_shared<Wait>(args));
                }

            }
        }
    }


    // create recv from each direction
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (0 != dx && 0 != dy) {

            Dim2<size_t> outbufExt(args_.nX + 2 * args_.nGhost, args_.nY + 2 * args_.nGhost);

            // recv into exterior
            Dim2<size_t> outbufOff;
            if (1 == dx) {
                outbufOff.x += args_.nX + args_.nGhost;
            }
            if (1 == dy) {
                outbufOff.y += args_.nY + args_.nGhost;
            }

            Dim2<size_t> unpackExt(args_.nX, args_.nY);
            if (0 != dx) {
                unpackExt.x = args_.nGhost;
            }
            if (0 != dy) {
                unpackExt.y = args_.nGhost;
            }



            // create unpack
            Unpack::Args unpackArgs;
            unpackArgs.outbuf = args_.grid;
            unpackArgs.pitch = args_.pitch;
            unpackArgs.nQ = args_.nQ;
            unpackArgs.outbufExt = outbufExt;
            unpackArgs.outbufOff = outbufOff;
            unpackArgs.unpackExt = unpackExt;
            unpackArgs.storageOrder = args_.storageOrder;
            auto unpack = std::make_shared<Unpack>(unpackArgs);

            const Dim2<size_t> srcCoord = myCoord + Dim2<size_t>(dx, dy);

            // create Irecv
            Irecv::Args recvArgs;
            recvArgs.buf = unpack->inbuf();
            recvArgs.count = unpackExt.x * unpackExt.y * args_.nQ;
            recvArgs.datatype = MPI_DOUBLE;
            recvArgs.source = args_.coordToRank(srcCoord);
            recvArgs.tag = 0;
            recvArgs.comm = MPI_COMM_WORLD;
            recvArgs.request = 0; // set by owner
            auto recv = std::make_shared<OwningIrecv>(recvArgs);
            recvs.push_back(recv);

            // connect preds -> Irecv
            for (auto &pred : preds) {
                g.then(pred, recv);
            }
            
            // connect Irecv -> unpack
            g.then(recv, unpack);

            // connect Irecv -> succs
            for (auto &succ : succs) {
                g.then(recv, succ);
            }

            // create a wait in waitRecvs
            {
                Wait::Args args;
                #warning skeleton
                waitRecvs.push_back(std::make_shared<Wait>(args));
            }
            }
        }
    }

    // all waitSends must wait for posts
    for (auto & wait : waitSends) {
        for (auto &send : sends) {
            g.then(send, wait);
        }
        for (auto &recv : recvs) {
            g.then(recv, wait);
        }
    }

    // all waitRecvs must wait for posts
    for (auto &wait : waitRecvs) {
        for (auto &send : sends) {
            g.then(send, wait);
        }
        for (auto &recv : recvs) {
            g.then(recv, wait);
        }
    }

    // remove this node from the graph
    g.erase(this);

}

/*
each warp covers a gridpoint, since quantities are stored consecutively
*/
__global__ void pack_kernel_qxy(
    double *outbuf,
    const double * inbuf,
    const Dim2<size_t> packExt,
    const Dim2<size_t> inbufOff,
    const size_t nQ,
    const size_t pitch
 ) {
    const int lx = threadIdx.x % 32;

    int warpsPerGridX = gridDim.x * blockDim.x / 32;

    for (size_t y = 0; y < packExt.y; y += gridDim.y * blockDim.y) {
        for (size_t x = 0; x < packExt.x; x += warpsPerGridX) {
            for (int q = lx; q < nQ; q += 32) {
                const double *ii = &inbuf[
                    (y + inbufOff.y) * pitch / sizeof(double) * nQ + 
                    (x + inbufOff.x) * nQ
                    + q
                ];
                double *oi = &outbuf[y * packExt.x * nQ + x * nQ + q];
                *oi = *ii;
            }
        }
    }

}

void Pack::run(cudaStream_t stream) {
    // each block does a 4x4 part of the grid
    switch(args_.storageOrder) {
        case HaloExchange::StorageOrder::QXY: {
            const int warps_x = 4;
            dim3 blockDim(warps_x * 32,4);
            dim3 gridDim(
                (args_.packExt.x + warps_x - 1) / warps_x,
                (args_.packExt.y + blockDim.y - 1) / blockDim.y
            );

            #define OR_THROW(b) {if (!(b)) throw std::runtime_error(AT);}
            OR_THROW(args_.inbuf);
            OR_THROW(outbuf_);
            #undef OR_THROW

            pack_kernel_qxy<<<gridDim, blockDim, 0, stream>>>(
                outbuf_.get(), args_.inbuf, args_.packExt, args_.inbufOff, args_.nQ, args_.pitch
            );
            break;
        }
        default:
        throw std::runtime_error(AT);
    }
}


/*
each warp covers a gridpoint, since quantities are stored consecutively
*/
__global__ void unpack_kernel_qxy(
    double *outbuf,
    const double * inbuf,
    const Dim2<size_t> unpackExt,
    const Dim2<size_t> outbufOff,
    const size_t nQ,
    const size_t pitch
 ) {
    const int lx = threadIdx.x % 32;

    int warpsPerGridX = gridDim.x * blockDim.x / 32;

    for (size_t y = 0; y < unpackExt.y; y += gridDim.y * blockDim.y) {
        for (size_t x = 0; x < unpackExt.x; x += warpsPerGridX) {
            for (int q = lx; q < nQ; q += 32) {
                double *oi = &outbuf[
                    (y + outbufOff.y) * pitch / sizeof(double) * nQ + 
                    (x + outbufOff.x) * nQ
                    + q
                ];
                const double *ii = &inbuf[y * unpackExt.x * nQ + x * nQ + q];
                *oi = *ii;
            }
        }
    }

}

void Unpack::run(cudaStream_t stream) {
    // each block does a 4x4 part of the grid
    switch(args_.storageOrder) {
        case HaloExchange::StorageOrder::QXY: {
            const int warps_x = 4;
            dim3 blockDim(warps_x * 32,4);
            dim3 gridDim(
                (args_.unpackExt.x + warps_x - 1) / warps_x,
                (args_.unpackExt.y + blockDim.y - 1) / blockDim.y
            );

            #define OR_THROW(b) {if (!(b)) throw std::runtime_error(AT);}
            OR_THROW(args_.outbuf);
            OR_THROW(inbuf_);
            #undef OR_THROW

            pack_kernel_qxy<<<gridDim, blockDim, 0, stream>>>(
                args_.outbuf, inbuf_.get(), args_.unpackExt, args_.outbufOff, args_.nQ, args_.pitch
            );
            break;
        }
        default:
        throw std::runtime_error(AT);
    }
}