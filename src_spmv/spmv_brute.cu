/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

/*! \file
 */

#include "tenzing/benchmarker.hpp"
#include "tenzing/brute.hpp"
#include "tenzing/cuda/cuda_runtime.hpp"
#include "tenzing/graph.hpp"
#include "tenzing/numeric.hpp"
#include "tenzing/schedule.hpp"
#include "tenzing/init.hpp"
#include "tenzing/spmv/csr_mat.hpp"
#include "tenzing/spmv/ops_spmv.cuh"
#include "tenzing/spmv/row_part_spmv.cuh"
#include "tenzing/spmv/where.hpp"

#include <mm/mm.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <thread>

#include <cusparse.h>

typedef int Ordinal;
typedef float Scalar;
typedef MtxReader<Ordinal, Scalar> reader_t;
typedef typename reader_t::coo_type mm_coo_t;
typedef typename reader_t::csr_type mm_csr_t;

template <Where w> using csr_type = CsrMat<w, Ordinal, Scalar>;

int main(int argc, char **argv) {

  tenzing::init();

  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Barrier(MPI_COMM_WORLD);

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

  CUDA_RUNTIME(cudaFree(0));

  /* interesting parameters:
     vortex: 1.5m rows, 15mnnz, bandwidth = 15m/16 4 nodes, 4 ranks per node
     may be even more interesting with 12 nodes, 4 ranks per node
  */
  int m = 150000;
  SpMV<Ordinal, Scalar>::Opts spmvOpts {
    .m = m,
    .bw = m/size,
    .nnz = m * 10
  };

  auto spmv = std::make_shared<SpMV<Ordinal, Scalar>>(spmvOpts, MPI_COMM_WORLD);

  Graph<OpBase> orig;
  orig.start_then(spmv);
  orig.then_finish(spmv);

  orig.dump();
  MPI_Barrier(MPI_COMM_WORLD);

  if (0 == rank) {
    std::cerr << "create platform";
  }
  Platform plat = Platform::make_n_streams(2, MPI_COMM_WORLD);

  EmpiricalBenchmarker benchmarker;

  brute::Opts opts;
  opts.benchOpts.nIters = 50;

  brute::Result result = brute::brute(orig, plat, benchmarker, opts);

  if (0 == rank)
    result.dump_csv();
}