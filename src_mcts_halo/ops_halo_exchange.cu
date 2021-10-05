#include "ops_halo_exchange.hpp"

#include "sched/ops_mpi.hpp"

#include "cuda_memory.hpp"

#define OR_THROW(b, msg) {if (!(b)) THROW_RUNTIME(msg)}

static int dir_to_tag(int dx, int dy, int dz) {
    if (dx < -1 || dx > 1) throw std::runtime_error(AT);
    if (dy < -1 || dy > 1) throw std::runtime_error(AT);
    if (dz < -1 || dz > 1) throw std::runtime_error(AT);
    dx += 1;// {-1,0,1} -> {0,1,2}
    dy += 1;
    dz += 1;
    return 9 * dz + 3 * dy + dx;
}

static bool exactly_one(bool b1, bool b2, bool b3) {
    return 
    (b1 && !b2 && !b3) ||
    (b2 && !b1 && !b3) ||
    (b3 && !b1 && !b2);
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

    const Dim3<int64_t> myCoord = args_.rankToCoord(rank);

    // create a single wait for all sends. This is the last thing done on the send side
    auto waitSend = std::make_shared<MultiWait>("he_wait_sends");
    for (auto &node : succs) {
        g.then(waitSend, node);
    }


    // create pack and send for each direction
    if (0 == rank) {std::cerr << "create sends\n";}
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (exactly_one(0 != dx, 0 != dy, 0 != dz)) {

                    Dim3<size_t> inbufExt(
                        args_.nX + 2 * args_.nGhost,
                        args_.nY + 2 * args_.nGhost,
                        args_.nZ + 2 * args_.nGhost
                    );

                    Dim3<size_t> inbufOff(args_.nGhost, args_.nGhost, args_.nGhost);
                    if (1 == dx) {
                        inbufOff.x += args_.nX;
                    }
                    if (1 == dy) {
                        inbufOff.y += args_.nY;
                    }
                    if (1 == dz) {
                        inbufOff.z += args_.nZ;
                    }

                    Dim3<size_t> packExt(args_.nX, args_.nY, args_.nZ);
                    if (0 != dx) {
                        packExt.x = args_.nGhost;
                    }
                    if (0 != dy) {
                        packExt.y = args_.nGhost;
                    }
                    if (0 != dz) {
                        packExt.z = args_.nGhost;
                    }



                    // create pack
                    std::stringstream packName;
                    packName << "he_pack_dx" << dx << "_dy" << dy << "_dz" << dz; 
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
                    const Dim3<int64_t> dstCoord = myCoord + Dim3<int64_t>(dx, dy, dz);

                    // create Isend
                    std::stringstream sendName;
                    sendName << "he_isend_dx" << dx << "_dy" << dy << "_dz" << dz; 
                    OwningIsend::Args sendArgs;
                    sendArgs.buf = pack->outbuf();
                    sendArgs.count = args_.nQ * packExt.x * packExt.y;
                    sendArgs.datatype = MPI_DOUBLE;
                    sendArgs.dest = args_.coordToRank(dstCoord);
                    sendArgs.tag = dir_to_tag(dx, dy, dz);
                    sendArgs.comm = MPI_COMM_WORLD;
                    sendArgs.request = nullptr; // will be set to owned req
                    auto send = std::make_shared<OwningIsend>(sendArgs, sendName.str());
                    sends.push_back(send);

                    waitSend->add_request(&send->request());


                    if (0 == rank) {
                        std::cerr << "send=<" << dx << "," << dy << "," << dz << "> "
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
    }


    // create recv from each direction
    if (0 == rank) {std::cerr << "create recvs\n";}
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (exactly_one(0 != dx, 0 != dy, 0 != dz)) {

                    Dim3<size_t> outbufExt(
                        args_.nX + 2 * args_.nGhost,
                        args_.nY + 2 * args_.nGhost,
                        args_.nZ + 2 * args_.nGhost
                    );

                    // recv into exterior
                    Dim3<size_t> outbufOff;
                    if (1 == dx) {
                        outbufOff.x += args_.nX + args_.nGhost;
                    }
                    if (1 == dy) {
                        outbufOff.y += args_.nY + args_.nGhost;
                    }
                    if (1 == dz) {
                        outbufOff.z += args_.nZ + args_.nGhost;
                    }

                    Dim3<size_t> unpackExt(args_.nX, args_.nY, args_.nZ);
                    if (0 != dx) {
                        unpackExt.x = args_.nGhost;
                    }
                    if (0 != dy) {
                        unpackExt.y = args_.nGhost;
                    }
                    if (0 != dz) {
                        unpackExt.z = args_.nGhost;
                    }

                    // create unpack
                    std::stringstream unpackName;
                    unpackName << "he_unpack_dx" << dx << "_dy" << dy << "_dz" << dz; 
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
                    const Dim3<int64_t> srcCoord = myCoord + Dim3<int64_t>(dx, dy, dz);

                    // create Irecv
                    std::stringstream recvName;
                    recvName << "he_irecv_dx" << dx << "_dy" << dy << "_dz" << dz; 
                    Irecv::Args recvArgs;
                    recvArgs.buf = unpack->inbuf();
                    recvArgs.count = unpackExt.x * unpackExt.y * args_.nQ;
                    recvArgs.datatype = MPI_DOUBLE;
                    recvArgs.source = args_.coordToRank(srcCoord);
                    recvArgs.tag = dir_to_tag(-dx, -dy, -dz); // reverse for send direction
                    recvArgs.comm = MPI_COMM_WORLD;
                    recvArgs.request = 0; // set by owner
                    auto recv = std::make_shared<OwningIrecv>(recvArgs, recvName.str());
                    recvs.push_back(recv);

                    if (0 == rank) {
                        std::cerr << "recv=<" << -dx << "," << -dy << "," << -dz << "> "
                        << "outbufExt=" << outbufExt
                        << " outbufOff=" << outbufOff
                        << " packExt=" << unpackExt
                        << " tag=" << recvArgs.tag
                        << std::endl;
                    }

                    std::stringstream waitName;
                    waitName << "he_waitrecv_dx" << dx << "_dy" << dy << "_dz" << dz; 

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


void HaloExchange::expand_3d_streams(
    Graph<Node> &g,
    cudaStream_t xStream,
    cudaStream_t yStream,
    cudaStream_t zStream
) {
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

    const Dim3<int64_t> myCoord = args_.rankToCoord(rank);

    // create a single wait for all sends. This is the last thing done on the send side
    auto waitSend = std::make_shared<MultiWait>("he_wait_sends");
    for (auto &node : succs) {
        g.then(waitSend, node);
    }


    // create pack and send for each direction
    if (0 == rank) {std::cerr << "create sends\n";}
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (exactly_one(0 != dx, 0 != dy, 0 != dz)) {

                    Dim3<size_t> inbufExt(
                        args_.nX + 2 * args_.nGhost,
                        args_.nY + 2 * args_.nGhost,
                        args_.nZ + 2 * args_.nGhost
                    );

                    Dim3<size_t> inbufOff(args_.nGhost, args_.nGhost, args_.nGhost);
                    if (1 == dx) {
                        inbufOff.x += args_.nX;
                    }
                    if (1 == dy) {
                        inbufOff.y += args_.nY;
                    }
                    if (1 == dz) {
                        inbufOff.z += args_.nZ;
                    }

                    Dim3<size_t> packExt(args_.nX, args_.nY, args_.nZ);
                    if (0 != dx) {
                        packExt.x = args_.nGhost;
                    }
                    if (0 != dy) {
                        packExt.y = args_.nGhost;
                    }
                    if (0 != dz) {
                        packExt.z = args_.nGhost;
                    }



                    // create pack
                    std::stringstream packName;
                    packName << "he_pack_dx" << dx << "_dy" << dy << "_dz" << dz; 
                    Pack::Args packArgs;
                    packArgs.inbufOff = inbufOff;
                    packArgs.packExt = packExt;
                    packArgs.inbufExt = inbufExt;
                    packArgs.pitch = args_.pitch;
                    packArgs.nQ = args_.nQ;
                    packArgs.storageOrder = args_.storageOrder;
                    packArgs.inbuf = args_.grid;
                    auto unassignedPack = std::make_shared<Pack>(packArgs, packName.str());
                    std::shared_ptr<StreamedOp> pack;
                    if (0 != dx) {
                        pack = std::make_shared<StreamedOp>(unassignedPack, xStream);
                    } else if (0 != dy) {
                        pack = std::make_shared<StreamedOp>(unassignedPack, yStream);
                    } else {
                        pack = std::make_shared<StreamedOp>(unassignedPack, zStream);
                    }
                    

                    // wrapping handled by rank conversion function
                    const Dim3<int64_t> dstCoord = myCoord + Dim3<int64_t>(dx, dy, dz);

                    // create Isend
                    std::stringstream sendName;
                    sendName << "he_isend_dx" << dx << "_dy" << dy << "_dz" << dz; 
                    OwningIsend::Args sendArgs;
                    sendArgs.buf = unassignedPack->outbuf();
                    sendArgs.count = args_.nQ * packExt.x * packExt.y;
                    sendArgs.datatype = MPI_DOUBLE;
                    sendArgs.dest = args_.coordToRank(dstCoord);
                    sendArgs.tag = dir_to_tag(dx, dy, dz);
                    sendArgs.comm = MPI_COMM_WORLD;
                    sendArgs.request = nullptr; // will be set to owned req
                    auto send = std::make_shared<OwningIsend>(sendArgs, sendName.str());
                    sends.push_back(send);

                    waitSend->add_request(&send->request());


                    if (0 == rank) {
                        std::cerr << "send=<" << dx << "," << dy << "," << dz << "> "
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
    }


    // create recv from each direction
    if (0 == rank) {std::cerr << "create recvs\n";}
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (exactly_one(0 != dx, 0 != dy, 0 != dz)) {

                    Dim3<size_t> outbufExt(
                        args_.nX + 2 * args_.nGhost,
                        args_.nY + 2 * args_.nGhost,
                        args_.nZ + 2 * args_.nGhost
                    );

                    // recv into exterior
                    Dim3<size_t> outbufOff;
                    if (1 == dx) {
                        outbufOff.x += args_.nX + args_.nGhost;
                    }
                    if (1 == dy) {
                        outbufOff.y += args_.nY + args_.nGhost;
                    }
                    if (1 == dz) {
                        outbufOff.z += args_.nZ + args_.nGhost;
                    }

                    Dim3<size_t> unpackExt(args_.nX, args_.nY, args_.nZ);
                    if (0 != dx) {
                        unpackExt.x = args_.nGhost;
                    }
                    if (0 != dy) {
                        unpackExt.y = args_.nGhost;
                    }
                    if (0 != dz) {
                        unpackExt.z = args_.nGhost;
                    }

                    // create unpack
                    std::stringstream unpackName;
                    unpackName << "he_unpack_dx" << dx << "_dy" << dy << "_dz" << dz; 
                    Unpack::Args unpackArgs;
                    unpackArgs.outbuf = args_.grid;
                    unpackArgs.pitch = args_.pitch;
                    unpackArgs.nQ = args_.nQ;
                    unpackArgs.outbufExt = outbufExt;
                    unpackArgs.outbufOff = outbufOff;
                    unpackArgs.unpackExt = unpackExt;
                    unpackArgs.storageOrder = args_.storageOrder;
                    auto unassignedUnpack = std::make_shared<Unpack>(unpackArgs, unpackName.str());
                    std::shared_ptr<StreamedOp> unpack;
                    if (0 != dx) {
                        unpack = std::make_shared<StreamedOp>(unassignedUnpack, xStream);
                    } else if (0 != dy) {
                        unpack = std::make_shared<StreamedOp>(unassignedUnpack, yStream);
                    } else {
                        unpack = std::make_shared<StreamedOp>(unassignedUnpack, zStream);
                    }

                    // wrapping handled by source conversion function
                    const Dim3<int64_t> srcCoord = myCoord + Dim3<int64_t>(dx, dy, dz);

                    // create Irecv
                    std::stringstream recvName;
                    recvName << "he_irecv_dx" << dx << "_dy" << dy << "_dz" << dz; 
                    Irecv::Args recvArgs;
                    recvArgs.buf = unassignedUnpack->inbuf();
                    recvArgs.count = unpackExt.x * unpackExt.y * args_.nQ;
                    recvArgs.datatype = MPI_DOUBLE;
                    recvArgs.source = args_.coordToRank(srcCoord);
                    recvArgs.tag = dir_to_tag(-dx, -dy, -dz); // reverse for send direction
                    recvArgs.comm = MPI_COMM_WORLD;
                    recvArgs.request = 0; // set by owner
                    auto recv = std::make_shared<OwningIrecv>(recvArgs, recvName.str());
                    recvs.push_back(recv);

                    if (0 == rank) {
                        std::cerr << "recv=<" << -dx << "," << -dy << "," << -dz << "> "
                        << "outbufExt=" << outbufExt
                        << " outbufOff=" << outbufOff
                        << " packExt=" << unpackExt
                        << " tag=" << recvArgs.tag
                        << std::endl;
                    }

                    std::stringstream waitName;
                    waitName << "he_waitrecv_dx" << dx << "_dy" << dy << "_dz" << dz; 

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
__global__ void pack_kernel_qxyz(
    double *outbuf,
    const double * inbuf,
    const Dim3<size_t> packExt,
    const Dim3<size_t> inbufOff,
    const Dim3<size_t> inbufExt,
    const size_t nQ,
    const size_t pitch
 ) {
    const int lx = threadIdx.x % 32;

    int warpsPerGridX = gridDim.x * blockDim.x / 32;

    for (size_t z = 0; z < packExt.z; z += gridDim.z * blockDim.z) {
        for (size_t y = 0; y < packExt.y; y += gridDim.y * blockDim.y) {
            for (size_t x = 0; x < packExt.x; x += warpsPerGridX) {
                for (int q = lx; q < nQ; q += 32) {

                    const size_t zi = z + inbufOff.z;
                    const size_t yi = y + inbufOff.y;
                    const size_t xi = x + inbufOff.x;
                    const size_t qi = q;

                    const double *ii = &inbuf[
                        zi * inbufExt.y * inbufExt.x * pitch / sizeof(double)
                        + yi * inbufExt.x * pitch / sizeof(double)
                        + xi * pitch / sizeof(double)
                        + qi
                    ];
                    double *oi = &outbuf[
                        z * packExt.y * packExt.x * nQ
                        + y * packExt.x * nQ
                        + x * nQ
                        + q
                    ];
                    *oi = *ii;
                }
            }
        }
    }
}

/*
each thread covers a gridpoint
*/
__global__ void pack_kernel_xyzq(
    double *outbuf,
    const double * inbuf,
    const Dim3<size_t> packExt,
    const Dim3<size_t> inbufOff,
    const Dim3<size_t> inbufExt,
    const size_t nQ,
    const size_t pitch
 ) {
    for (int q = 0; q < nQ; ++q) {
        for (size_t z = 0; z < packExt.z; z += gridDim.z * blockDim.z) {
            for (size_t y = 0; y < packExt.y; y += gridDim.y * blockDim.y) {
                for (size_t x = 0; x < packExt.x; x += gridDim.x * blockDim.x) {

                    const size_t qi = q;
                    const size_t zi = z + inbufOff.z;
                    const size_t yi = y + inbufOff.y;
                    const size_t xi = x + inbufOff.x;

                    const double *ii = &inbuf[
                        qi * inbufExt.z * inbufExt.y * pitch / sizeof(double)
                        + zi * inbufExt.y * pitch / sizeof(double)
                        + yi * pitch / sizeof(double)
                        + xi
                    ];
                    double *oi = &outbuf[
                        q * packExt.z * packExt.y * packExt.x
                        + z * packExt.y * packExt.x
                        + y * packExt.x
                        + x
                    ];
                    *oi = *ii;
                }
            }
        }
    }
}

void Pack::run(cudaStream_t stream) {

    OR_THROW(args_.inbuf, "Pack operation " << name() << " with null input buffer");
    OR_THROW(outbuf_, "Pack operation " << name() << " with null output buffer");

    switch(args_.storageOrder) {
        case HaloExchange::StorageOrder::QXYZ: {
            // each block does a 4x4 part of the grid
            const int warps_x = 4;
            dim3 blockDim(warps_x * 32,2,2);
            dim3 gridDim(
                (args_.packExt.x + warps_x - 1) / warps_x,
                (args_.packExt.y + blockDim.y - 1) / blockDim.y,
                (args_.packExt.z + blockDim.z - 1) / blockDim.z
            );
            pack_kernel_qxyz<<<gridDim, blockDim, 0, stream>>>(
                outbuf_.get(), args_.inbuf, args_.packExt, args_.inbufOff, args_.inbufExt, args_.nQ, args_.pitch
            );
            break;
        }
        case HaloExchange::StorageOrder::XYZQ: {
            dim3 blockDim(32,4,4);
            dim3 gridDim(
                (args_.packExt.x + blockDim.x - 1) / blockDim.x,
                (args_.packExt.y + blockDim.y - 1) / blockDim.y,
                (args_.packExt.z + blockDim.z - 1) / blockDim.z
            );
            pack_kernel_xyzq<<<gridDim, blockDim, 0, stream>>>(
                outbuf_.get(), args_.inbuf, args_.packExt, args_.inbufOff, args_.inbufExt, args_.nQ, args_.pitch
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
__global__ void unpack_kernel_qxyz(
    double *outbuf,
    const double * inbuf,
    const Dim3<size_t> unpackExt,
    const Dim3<size_t> outbufOff,
    const Dim3<size_t> outbufExt,
    const size_t nQ,
    const size_t pitch
 ) {
    const int lx = threadIdx.x % 32;

    int warpsPerGridX = gridDim.x * blockDim.x / 32;

    for (size_t z = 0; z < unpackExt.z; z += gridDim.z * blockDim.z) {
        for (size_t y = 0; y < unpackExt.y; y += gridDim.y * blockDim.y) {
            for (size_t x = 0; x < unpackExt.x; x += warpsPerGridX) {
                for (int q = lx; q < nQ; q += 32) {
                    double *oi = &outbuf[
                        (z + outbufOff.z) * outbufExt.y * outbufExt.x * pitch / sizeof(double)
                        +(y + outbufOff.y) * outbufExt.x * pitch / sizeof(double)
                        + (x + outbufOff.x) * pitch / sizeof(double)
                        + q
                    ];
                    const double *ii = &inbuf[
                        z * unpackExt.y * unpackExt.x  * nQ
                        + y * unpackExt.x * nQ 
                        + x * nQ
                        + q
                    ];
                    *oi = *ii;
                }
            }
        }
    }
}

/*
one thread per gridpoint
*/
__global__ void unpack_kernel_xyzq(
    double *outbuf,
    const double * inbuf,
    const Dim3<size_t> unpackExt,
    const Dim3<size_t> outbufOff,
    const Dim3<size_t> outbufExt,
    const size_t nQ,
    const size_t pitch
 ) {
    for (size_t q = 0; q < nQ; ++q) {
        for (size_t z = 0; z < unpackExt.z; z += gridDim.z * blockDim.z) {
            for (size_t y = 0; y < unpackExt.y; y += gridDim.y * blockDim.y) {
                for (size_t x = 0; x < unpackExt.x; x += gridDim.x * blockDim.x) {

                    const size_t qi = q;
                    const size_t zi = z + outbufOff.z;
                    const size_t yi = y + outbufOff.y;
                    const size_t xi = x + outbufOff.x;

                    double *oi = &outbuf[
                        qi * outbufExt.z * outbufExt.y * pitch / sizeof(double)
                        + zi * outbufExt.y * pitch / sizeof(double)
                        + yi * pitch / sizeof(double)
                        + xi
                    ];
                    const double *ii = &inbuf[
                        q * unpackExt.z * unpackExt.y * unpackExt.x
                        + z * unpackExt.y * unpackExt.x
                        + y * unpackExt.x
                        + x
                    ];

                    *oi = *ii;
                }
            }
        }
    }
}

void Unpack::run(cudaStream_t stream) {
    // each block does a 4x4 part of the grid

    OR_THROW(args_.outbuf, AT);
    OR_THROW(inbuf_, AT);

    switch(args_.storageOrder) {
        case HaloExchange::StorageOrder::QXYZ: {
            const int warps_x = 4;
            dim3 blockDim(warps_x * 32,2,2);
            dim3 gridDim(
                (args_.unpackExt.x + warps_x - 1) / warps_x,
                (args_.unpackExt.y + blockDim.y - 1) / blockDim.y,
                (args_.unpackExt.z + blockDim.z - 1) / blockDim.z
            );


            unpack_kernel_qxyz<<<gridDim, blockDim, 0, stream>>>(
                args_.outbuf, inbuf_.get(), args_.unpackExt, args_.outbufOff, args_.outbufExt, args_.nQ, args_.pitch
            );
            break;
        }
        case HaloExchange::StorageOrder::XYZQ: {
            dim3 blockDim(32,4,4);
            dim3 gridDim(
                (args_.unpackExt.x + blockDim.x - 1) / blockDim.x,
                (args_.unpackExt.y + blockDim.y - 1) / blockDim.y,
                (args_.unpackExt.z + blockDim.z - 1) / blockDim.z
            );
            unpack_kernel_xyzq<<<gridDim, blockDim, 0, stream>>>(
                args_.outbuf, inbuf_.get(), args_.unpackExt, args_.outbufOff, args_.outbufExt, args_.nQ, args_.pitch
            );
            CUDA_RUNTIME(cudaDeviceSynchronize());
            break;
        }
        default:
        throw std::runtime_error(AT);
    }
}

#undef OR_THROW