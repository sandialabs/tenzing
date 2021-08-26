#include "ops_halo_exchange.hpp"

#include "sched/ops_mpi.hpp"

#include "cuda_memory.hpp"
/*
dx | dy | tag
-1 | -1 |   0
-1 |  0 |   1
-1 |  1 |   2
 0 | -1 |   3
 0 |  0 |   4
 0 |  1 |   5
 1 | -1 |   6
 1 |  0 |   7
 1 |  1 |   8
*/
static int dir_to_tag(int dx, int dy) {
    if (dx < -1 || dx > 1) throw std::runtime_error(AT);
    if (dy < -1 || dy > 1) throw std::runtime_error(AT);
    dx += 1;// {-1,0,1} -> {0,1,2}
    dy += 1;
    return 3 * dx + dy;
}

void HaloExchange::expand_in(Graph<Node> &g) {
    // find preds and successors of this graph
    Graph<Node>::NodeSet &preds = g.preds(this);
    Graph<Node>::NodeSet &succs = g.succs(this);

    // new nodes created to replace this node
    std::vector<std::shared_ptr<Isend>> sends;
    std::vector<std::shared_ptr<Irecv>> recvs;
    std::vector<std::shared_ptr<Wait>> waitRecvs;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const Dim2<int64_t> myCoord = args_.rankToCoord(rank);

    // create a single wait for all sends. This is the last thing done on the send side
    auto waitSend = std::make_shared<MultiWait>("he_wait_sends");
    for (auto &node : succs) {
        g.then(waitSend, node);
    }


    // create pack and send for each direction
    if (0 == rank) {std::cerr << "create sends\n";}
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (0 == dx ^ 0 == dy) {

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
                std::stringstream packName;
                packName << "he_pack_dx" << dx << "_dy" << dy; 
                Pack::Args packArgs;
                packArgs.inbufOff = inbufOff;
                packArgs.packExt = packExt;
                packArgs.inbufExt = inbufExt;
                packArgs.pitch = args_.pitch;
                packArgs.nQ = args_.nQ;
                packArgs.storageOrder = args_.storageOrder;
                packArgs.inbuf = args_.grid;
                auto pack = std::make_shared<Pack>(packArgs, packName.str());

                // wrapping handled by rank conversion function
                const Dim2<int64_t> dstCoord = myCoord + Dim2<int64_t>(dx, dy);

                // create Isend
                std::stringstream sendName;
                sendName << "he_isend_dx" << dx << "_dy" << dy; 
                OwningIsend::Args sendArgs;
                sendArgs.buf = pack->outbuf();
                sendArgs.count = args_.nQ * packExt.x * packExt.y;
                sendArgs.datatype = MPI_DOUBLE;
                sendArgs.dest = args_.coordToRank(dstCoord);
                sendArgs.tag = dir_to_tag(dx, dy);
                sendArgs.comm = MPI_COMM_WORLD;
                sendArgs.request = nullptr; // will be set to owned req
                auto send = std::make_shared<OwningIsend>(sendArgs, sendName.str());
                sends.push_back(send);

                waitSend->add_request(&send->request());


                if (0 == rank) {
                    std::cerr << "send=<" << dx << "," << dy << "> "
                    << "inbufExt=" << inbufExt
                    << " inbufOff=" << inbufOff
                    << " packExt=" << packExt
                    << " tag=" << sendArgs.tag
                    << std::endl;
                }
                

                if (0 == rank) {std::cerr << "connect preds -> pack\n";}
                for (auto &pred : preds) {
                    g.then(pred, pack);
                }
                
                if (0 == rank) {std::cerr << "connect pack -> Isend\n";}
                g.then(pack, send);
            
                if (0 == rank) {std::cerr << "connect Isend -> waitSend\n";}
                g.then(send, waitSend);

                #if 0
                if (0 == rank) {std::cerr << "connect Isend -> waitall\n";}
                for (auto &succ : succs) {
                    g.then(wait, succ);
                }
                #endif


            }
        }
    }


    // create recv from each direction
    if (0 == rank) {std::cerr << "create recvs\n";}
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (0 == dx ^ 0 == dy) {

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
                std::stringstream unpackName;
                unpackName << "he_unpack_dx" << dx << "_dy" << dy; 
                Unpack::Args unpackArgs;
                unpackArgs.outbuf = args_.grid;
                unpackArgs.pitch = args_.pitch;
                unpackArgs.nQ = args_.nQ;
                unpackArgs.outbufExt = outbufExt;
                unpackArgs.outbufOff = outbufOff;
                unpackArgs.unpackExt = unpackExt;
                unpackArgs.storageOrder = args_.storageOrder;
                auto unpack = std::make_shared<Unpack>(unpackArgs, unpackName.str());

                // wrapping handled by source conversion function
                const Dim2<int64_t> srcCoord = myCoord + Dim2<int64_t>(dx, dy);

                // create Irecv
                std::stringstream recvName;
                recvName << "he_irecv_dx" << dx << "_dy" << dy; 
                Irecv::Args recvArgs;
                recvArgs.buf = unpack->inbuf();
                recvArgs.count = unpackExt.x * unpackExt.y * args_.nQ;
                recvArgs.datatype = MPI_DOUBLE;
                recvArgs.source = args_.coordToRank(srcCoord);
                recvArgs.tag = dir_to_tag(-dx, -dy); // reverse for send direction
                recvArgs.comm = MPI_COMM_WORLD;
                recvArgs.request = 0; // set by owner
                auto recv = std::make_shared<OwningIrecv>(recvArgs, recvName.str());
                recvs.push_back(recv);

                if (0 == rank) {
                    std::cerr << "recv=<" << -dx << "," << -dy << "> "
                    << "outbufExt=" << outbufExt
                    << " outbufOff=" << outbufOff
                    << " packExt=" << unpackExt
                    << " tag=" << recvArgs.tag
                    << std::endl;
                }

                std::stringstream waitName;
                waitName << "he_waitrecv_dx" << dx << "_dy" << dy; 

                // create a wait in waitRecvs
                Wait::Args waitArgs;
                waitArgs.request = &recv->request();
                waitArgs.status = MPI_STATUS_IGNORE;
                auto wait = std::make_shared<Wait>(waitArgs, waitName.str());
                waitRecvs.push_back(wait);

                // connect preds -> Irecv
                for (auto &pred : preds) {
                    g.then(pred, recv);
                }
                
                // connect Irecv -> wait -> unpack
                g.then(recv, wait);
                g.then(wait, unpack);

                // connect unpack -> succs
                for (auto &succ : succs) {
                    g.then(unpack, succ);
                }

                // waitSend must wait for all posts
                g.then(recv, waitSend);
            }
        }
    }

    // all waitRecvs must wait for all posts
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

            #define OR_THROW(b, msg) {if (!(b)) THROW_RUNTIME(msg)}
            OR_THROW(args_.inbuf, "Pack operation " << name() << " with null input buffer");
            OR_THROW(outbuf_, "Pack operation " << name() << " with null output buffer");
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