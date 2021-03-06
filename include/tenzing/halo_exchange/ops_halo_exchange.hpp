/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include "dim.hpp"

#include "tenzing/halo_exchange/cuda_memory.hpp"
#include "tenzing/graph.hpp"
#include "tenzing/operation.hpp"
#include "tenzing/mpi/ops_mpi.hpp"

#include <functional>

/*
  A 3D halo exchange

  Each direction is packed and sent separately
*/
class HaloExchange {

public:
  // ABC: in linear memory, A increments fastest, then B then C
  enum class StorageOrder {
    QXYZ, // Q increments fastest (Q0, Q1... stored for X0Y0, then Q0,Q1,... for X1Y0...
    XYZQ, // X increments fastest, then Y then Q
  };

  // typedef Dim2<int64_t> (*RankToCoordFn)(int rank); // 2D coordinate for each rank
  // typedef int (*CoordToRankFn)(const Dim2<int64_t> &coord); // rank for each 2D coordinate
  typedef std::function<Dim3<int64_t>(int)> RankToCoordFn;
  typedef std::function<int(const Dim3<int64_t> &)> CoordToRankFn;

  struct Args {
    RankToCoordFn rankToCoord;
    CoordToRankFn coordToRank;
    StorageOrder storageOrder;
    size_t pitch;
    size_t nX; // grid size x (no ghost)
    size_t nY; // grid size y (no ghost)
    size_t nZ; // grid size z (no ghost)
    size_t nQ;
    size_t nGhost;
    double *grid;

    Args() : rankToCoord(nullptr), coordToRank(nullptr), grid(nullptr) {}
    bool operator==(const Args &rhs) const {
#define FEQ(x) (x == rhs.x)
      return FEQ(storageOrder) && FEQ(pitch) && FEQ(nX) && FEQ(nY) && FEQ(nQ) && FEQ(nGhost) &&
             FEQ(grid);
#undef FEQ
    }
  };

  // access grid value at local gridpoint x,y,q (caller must handle ghost cells)
  template <StorageOrder SO> static double &at(double *p, size_t x, size_t y, size_t q) {}

  // add the halo exchange to a graph, using preds as preds and succs as successors
  static void add_to_graph(Graph<OpBase> &g, const Args &args,
                           const std::vector<std::shared_ptr<OpBase>> &preds,
                           const std::vector<std::shared_ptr<OpBase>> &succs);
};

/* like an Isend, but owns its request
 */
class OwningIsend : public Isend {
private:
protected:
  MPI_Request req_;

public:
  OwningIsend(const Args &args, const std::string &name) : Isend(args, name) {
    args_.request = &req_;
  }
  MPI_Request &request() { return req_; }
};

/* like an Irecv, but owns its request
 */
class OwningIrecv : public Irecv {
private:
protected:
  MPI_Request req_;

public:
  OwningIrecv(const Args &args, const std::string &name) : Irecv(args, name) {
    args_.request = &req_;
  }
  MPI_Request &request() { return req_; }
};

/* packs a 2D region into a buffer
   owns its output buffer
*/
class Pack : public GpuOp {
public:
  /* inbuf must live at least as long as pack
   */
  struct Args {
    const double *inbuf;
    size_t pitch;
    size_t nQ;
    Dim3<size_t> inbufExt; // size of the input buffer (elements)
    Dim3<size_t> inbufOff; // offset into the input buffer
    Dim3<size_t> packExt;  // size of the region to copy
    HaloExchange::StorageOrder storageOrder;

    bool operator==(const Args &rhs) const {
#define FEQ(x) (x == rhs.x)
      return FEQ(inbuf) && FEQ(inbufExt) && FEQ(inbufOff) && FEQ(packExt);
#undef FEQ
    }
  };

private:
  Args args_;
  std::string name_;
  std::shared_ptr<double> outbuf_;

public:
  Pack(const Args &args, const std::string &name) : args_(args), name_(name) {
    outbuf_ =
        cuda_make_shared<double>(args_.nQ * args_.packExt.x * args_.packExt.y * args_.packExt.z);
  }

  // Node functions
  std::string name() const override { return name_; }
  EQ_DEF(Pack);
  LT_DEF(Pack);
  CLONE_DEF(Pack);
  bool operator<(const Pack &rhs) const { return name() < rhs.name(); }
  bool operator==(const Pack &rhs) const { return args_ == rhs.args_; }

  virtual void run(cudaStream_t stream) override;

  const double *outbuf() const { return outbuf_.get(); }
};

/* unpacks a buffer into a 2D region
   owns its input buffer
*/
class Unpack : public GpuOp {
public:
  /* inbuf must live at least as long as pack
   */
  struct Args {
    double *outbuf;
    size_t pitch; // pitch of output buffer
    size_t nQ;
    Dim3<size_t> outbufExt; // size of the output buffer (elements)
    Dim3<size_t> outbufOff; // offset into the input buffer
    Dim3<size_t> unpackExt; // size of the region to copy
    HaloExchange::StorageOrder storageOrder;

    bool operator==(const Args &rhs) const {
#define FEQ(x) (x == rhs.x)
      return FEQ(outbuf) && FEQ(outbufExt) && FEQ(outbufOff) && FEQ(unpackExt);
#undef FEQ
    }
  };

private:
  Args args_;
  std::string name_;
  std::shared_ptr<double> inbuf_;

public:
  Unpack(const Args &args, const std::string &name) : args_(args), name_(name) {
    inbuf_ = cuda_make_shared<double>(args_.nQ * args_.unpackExt.x * args_.unpackExt.y *
                                      args.unpackExt.z);
  }

  // Node functions
  std::string name() const override { return name_; }
  EQ_DEF(Unpack);
  LT_DEF(Unpack);
  CLONE_DEF(Unpack);
  bool operator<(const Unpack &rhs) const { return name() < rhs.name(); }
  bool operator==(const Unpack &rhs) const { return args_ == rhs.args_; }

  virtual void run(cudaStream_t stream) override;

  double *inbuf() const { return inbuf_.get(); }
};