#pragma once

#include "dim.hpp"
#include "cuda_memory.hpp"

#include "sched/operation.hpp"
#include "sched/ops_mpi.hpp"
#include "sched/graph.hpp"

class Expandable {
    public:
    virtual void expand_in(Graph<Node> &g) = 0;
};

/*
  A 2D halo exchange
*/
class HaloExchange : public CpuNode, public Expandable {

public:

    // ABC: in linear memory, A increments fastest, then B then C
    enum class StorageOrder {
        QXY, // Q increments fastest (Q0, Q1... stored for X0Y0, then Q0,Q1,... for X1Y0...
    };

    typedef Dim2<size_t> (*RankToCoordFn)(int rank); // 2D coordinate for each rank
    typedef int (*CoordToRankFn)(const Dim2<size_t> &coord); // rank for each 2D coordinate


struct Args {
    RankToCoordFn rankToCoord;
    CoordToRankFn  coordToRank;
    StorageOrder storageOrder;
    size_t pitch;
    size_t nX; // grid size x (no ghost)
    size_t nY; // grid size y (no ghost)
    size_t nQ;
    size_t nGhost;
    double *grid;

    Args() : rankToCoord(nullptr), coordToRank(nullptr), grid(nullptr) {}
    bool operator==(const Args &rhs) const {
        #define FEQ(x) (x == rhs.x)
        return FEQ(rankToCoord)
        && FEQ(coordToRank)
        && FEQ(storageOrder)
        && FEQ(pitch)
        && FEQ(nX)
        && FEQ(nY)
        && FEQ(nQ)
        && FEQ(nGhost)
        && FEQ(grid);
        #undef FEQ
    }
};



private:
    Args args_;

public:

    HaloExchange(const Args &args) : args_(args) {}


    // access grid value at local gridpoint x,y,q (caller must handle ghost cells)
    template<StorageOrder SO>
    double &at(double *p, size_t x, size_t y, size_t q);

    // Node functions
    std::string name() const override { return "HaloExchange"; }
    EQ_DEF(HaloExchange);
    LT_DEF(HaloExchange);
    CLONE_DEF(HaloExchange);
    virtual int tag() const override { return 5; }
    bool operator<(const HaloExchange &rhs) const {return name() < rhs.name(); }
    bool operator==(const HaloExchange &rhs) const {return args_ == rhs.args_; }

    // expander functions
    virtual void expand_in(Graph<Node> &g) override;
};

template<>
inline double &HaloExchange::at<HaloExchange::StorageOrder::QXY>(double *p, size_t x, size_t y, size_t q) {
    const size_t extX = args_.nX + 2 * args_.nGhost;
    return p[y * extX * args_.pitch + x * args_.pitch + q];
}


/* like an Isend, but owns its request
*/
class OwningIsend : public Isend {
private:

protected:
    MPI_Request req_;

public:
    OwningIsend(const Args &args) : Isend(args) {
        args_.request = &req_;
    }
};

/* like an Irecv, but owns its request
*/
class OwningIrecv : public Irecv {
private:

protected:
    MPI_Request req_;
    
public:
    OwningIrecv(const Args &args) : Irecv(args) {
        args_.request = &req_;
    }
};

/* packs a 2D region into a buffer
   owns its output buffer
*/
class Pack : public GpuNode {
public:
    /* inbuf must live at least as long as pack
    */
    struct Args {
        const double *inbuf;
        size_t pitch;
        size_t nQ;
        Dim2<size_t> inbufExt; // size of the input buffer (elements)
        Dim2<size_t> inbufOff; // offset into the input buffer
        Dim2<size_t> packExt; // size of the region to copy
        HaloExchange::StorageOrder storageOrder;

        bool operator==(const Args &rhs) const {
            #define FEQ(x) (x == rhs.x)
            return FEQ(inbuf)
            && FEQ(inbufExt)
            && FEQ(inbufOff)
            && FEQ(packExt);
            #undef FEQ
        }
    };

private:
    Args args_;
    std::shared_ptr<double> outbuf_;
public:
    Pack(const Args &args) : args_(args) {
        outbuf_ = cuda_make_shared<double>(args_.nQ * args_.packExt.x * args_.packExt.y);
    }

    // Node functions
    std::string name() const override { return "Pack"; }
    EQ_DEF(Pack);
    LT_DEF(Pack);
    CLONE_DEF(Pack);
    virtual int tag() const override { return 6; }
    bool operator<(const Pack &rhs) const {
        #warning fixme
    return name() < rhs.name(); }
    bool operator==(const Pack &rhs) const {return args_ == rhs.args_; }

    virtual void run(cudaStream_t stream) override;

    const double *outbuf() const {return outbuf_.get(); }
};


/* unpacks a buffer into a 2D region
   owns its input buffer
*/
class Unpack : public GpuNode {
public:
    /* inbuf must live at least as long as pack
    */
    struct Args {
        double *outbuf;
        size_t pitch; // pitch of output buffer
        size_t nQ;
        Dim2<size_t> outbufExt; // size of the output buffer (elements)
        Dim2<size_t> outbufOff; // offset into the input buffer
        Dim2<size_t> unpackExt; // size of the region to copy
        HaloExchange::StorageOrder storageOrder;

        bool operator==(const Args &rhs) const {
            #define FEQ(x) (x == rhs.x)
            return FEQ(outbuf)
            && FEQ(outbufExt)
            && FEQ(outbufOff)
            && FEQ(unpackExt);
            #undef FEQ
        }
    };

private:
    Args args_;
    std::shared_ptr<double> inbuf_;
public:
    Unpack(const Args &args) : args_(args) {
        inbuf_ = cuda_make_shared<double>(args_.nQ * args_.unpackExt.x * args_.unpackExt.y);
    }

    // Node functions
    std::string name() const override { return "Unpack"; }
    EQ_DEF(Unpack);
    LT_DEF(Unpack);
    CLONE_DEF(Unpack);
    virtual int tag() const override { return 7; }
    bool operator<(const Unpack &rhs) const {
        #warning fixme
        return name() < rhs.name(); }
    bool operator==(const Unpack &rhs) const {return args_ == rhs.args_; }

    virtual void run(cudaStream_t stream) override;

    double *inbuf() const {return inbuf_.get(); }
};