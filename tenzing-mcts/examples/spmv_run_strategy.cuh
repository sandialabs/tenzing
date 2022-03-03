/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

/* use MCTS for both order and streams
 */

#include <argparse/argparse.hpp>

#include "tenzing/benchmarker.hpp"
#include "tenzing/cuda/cuda_runtime.hpp"
#include "tenzing/graph.hpp"
#include "tenzing/init.hpp"
#include "tenzing/numeric.hpp"
#include "tenzing/platform.hpp"
#include "tenzing/schedule.hpp"
#include "tenzing/spmv/ops_spmv.cuh"
#include "tenzing/reproduce.hpp"

#include "tenzing/mcts/mcts.hpp"

typedef int Ordinal;
typedef float Scalar;

template <Where w> using csr_type = CsrMat<w, Ordinal, Scalar>;

template <typename Strategy> int doit(tenzing::mcts::Opts &opts, int argc, char **argv) {

  tenzing::init(argc, argv);

  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (0 == rank) {
    tenzing::reproduce::dump_with_cli(argc, argv);
  }



  opts.nIters = 300;
  opts.benchOpts.nIters = 50;

  int m = 150000; // matrix size

  bool noExpandRollout = false;
  argparse::Parser parser("SpMV design-space exporation using monte-carlo tree search");
  parser.add_option(opts.nIters, "--mcts-iters", "-i")->help("how many MCTS iterations to do");
  parser.add_option(opts.benchOpts.nIters, "--benchmark-iters", "-b")
      ->help("how many benchmark measurements to do.");
  parser.add_option(m, "--matrix-m", "-m")->help("random matrix dimension");
  parser.add_flag(noExpandRollout, "--no-expand-rollout")->help("don't expand rollout");
  parser.no_unrecognized();

  if (!parser.parse(argc, argv)) {
    std::cerr << parser.help();
    exit(EXIT_FAILURE);
  }
  opts.expandRollout = !noExpandRollout;




  int bw = m / size;
  int nnz = m * 10;

  {
    char hostname[MPI_MAX_PROCESSOR_NAME] = {};
    int len;
    MPI_Get_processor_name(hostname, &len);

    const char *p = std::getenv("OMP_PLACES");
    if (!p)
      p = "<unset>";
    std::cerr << "rank " << rank << " of " << size << " on " << hostname << " OMP_PLACES: " << p
              << "\n";

    // round-robin GPU scheduling
    int devcount;
    CUDA_RUNTIME(cudaGetDeviceCount(&devcount));
    int dev = rank % devcount;
    CUDA_RUNTIME(cudaSetDevice(dev));

    cudaDeviceProp prop;
    CUDA_RUNTIME(cudaGetDeviceProperties(&prop, dev));
    fprintf(stderr, "rank %d on %s GPU %08x:%02x:%02x.0 (%d)\n", rank, hostname, prop.pciDomainID,
            prop.pciBusID, prop.pciDeviceID, dev);
  }

  EmpiricalBenchmarker benchmarker;

  MPI_Barrier(MPI_COMM_WORLD);

  /* interesting parameters:
     vortex: 1.5m rows, 15mnnz, bandwidth = 15m/16 4 nodes, 4 ranks per node
     may be even more interesting with 12 nodes, 4 ranks per node
  */

  csr_type<Where::host> A;

  // generate and distribute A
  if (0 == rank) {
    std::cerr << "generate matrix\n";
    A = random_band_matrix<Ordinal, Scalar>(m, bw, nnz);
  }

  RowPartSpmv<Ordinal, Scalar> rps(A, 0, MPI_COMM_WORLD);

  auto spmv = std::make_shared<SpMV<Ordinal, Scalar>>(rps, MPI_COMM_WORLD);


  Graph<OpBase> orig;
  orig.start_then(spmv);
  orig.then_finish(spmv);

  orig.dump();
  MPI_Barrier(MPI_COMM_WORLD);

  if (0 == rank) {
    std::cerr << "create platform";
  }
  Platform platform = Platform::make_n_streams(2, MPI_COMM_WORLD);

  STDERR("mcts...");

  tenzing::mcts::Result result = tenzing::mcts::explore<Strategy>(orig, platform, benchmarker, opts);
  if (0 == rank)
    result.dump_csv();

  return 0;
}