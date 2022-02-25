/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "tenzing/mcts.hpp"
#include "tenzing/numeric.hpp"
#include "tenzing/operation.hpp"
#include "tenzing/schedule.hpp"
#include "tenzing/init.hpp"
#include "tenzing/halo_exchange/ops_halo_exchange.hpp"

#include <mpi.h>

#include <algorithm>
#include <memory>
#include <vector>

template <typename Strategy> int doit(int argc, char **argv) {
  (void)argc;
  (void)argv;


  typedef HaloExchange::StorageOrder StorageOrder;
  typedef HaloExchange::Args Args;

  tenzing::init();

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  typedef double Real;

  Args args;
  args.nQ = 3;   // quantities per gridpoint
  args.nX = 512; // x and y extent of cells / rank
  args.nY = 512;
  args.nZ = 512;
  args.pitch = 128; // pitch of allocated memory in bytes
  args.nGhost = 3;  // ghost cell radius
  args.storageOrder = StorageOrder::XYZQ;

  /* allocate width * height * depth
   */
  {
    size_t pitch, d2, d3, d4;
    switch (args.storageOrder) {
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

    std::cerr << "alloc p=" << pitch << " d2=" << d2 << " d3=" << d3 << " d4=" << d4 << " ("
              << pitch * d2 * d3 * d4 / 1024.0 / 1024.0 << "MiB)\n";
    CUDA_RUNTIME(cudaMalloc(&args.grid, pitch * d2 * d3 * d4));
  }

  // rank dimensions
  Dim3<int64_t> rd(1, 1, 1);

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
    if (0 == rank)
      std::cerr << "rank grid: " << rd << "\n";
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
    while (wrapped.x < 0) {
      wrapped.x += rd.x;
    }
    while (wrapped.y < 0) {
      wrapped.y += rd.y;
    }
    while (wrapped.z < 0) {
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

  std::cerr << "create graph\n";
  Graph<OpBase> orig;
  HaloExchange::add_to_graph(orig, args, {orig.start()}, {orig.finish()});

  if (0 == rank) {
    orig.dump_graphviz("orig.dot");
  }

  if (0 == rank) {
    std::cerr << "dump\n";
    orig.dump_graphviz("expanded.dot");
  }

  if (0 == rank) {
    std::cerr << "create platform";
  }
  Platform platform = Platform::make_n_streams(2, MPI_COMM_WORLD);
  EmpiricalBenchmarker benchmarker;


  mcts::Opts opts;
  opts.benchOpts.nIters = 50;
  opts.dumpTreePrefix = "halo";

  STDERR("mcts...");
  mcts::Result result = mcts::mcts<Strategy>(orig, platform, benchmarker, opts);

  result.dump_csv();

  return 0;
}