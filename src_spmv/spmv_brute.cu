/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

/*! \file
 */

#include "sched/benchmarker.hpp"
#include "sched/brute.hpp"
#include "sched/cuda_runtime.hpp"
#include "sched/graph.hpp"
#include "sched/numeric.hpp"
#include "sched/schedule.hpp"
#include "sched/init.hpp"
#include "sched/spmv/csr_mat.hpp"
#include "sched/spmv/ops_spmv.cuh"
#include "sched/spmv/row_part_spmv.cuh"
#include "sched/spmv/where.hpp"

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

  sched::init();

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

  /* interesting parameters:
     vortex: 1.5m rows, 15mnnz, bandwidth = 15m/16 4 nodes, 4 ranks per node
     may be even more interesting with 12 nodes, 4 ranks per node
  */

  int m = 150000;
  int bw = m / size;
  int nnz = m * 10;

  csr_type<Where::host> A;

  if (argc < 2) {

    // generate and distribute A
    if (0 == rank) {
      std::cerr << "generate matrix\n";
      A = random_band_matrix<Ordinal, Scalar>(m, bw, nnz);
    }
  } else {
    if (0 == rank) {
      std::string path = argv[1];
      std::cerr << "load " << path << std::endl;
      reader_t reader(path);
      mm_coo_t coo = reader.read_coo();
      mm_csr_t csr(coo);

      std::cerr << "allocate A\n";
      A = csr_type<Where::host>(csr.num_rows(), csr.num_cols(), csr.nnz());

      std::cerr << "move CSR data...\n";
      for (size_t i = 0; i < csr.row_ptr().size(); ++i) {
        A.row_ptr()[i] = csr.row_ptr(i);
      }
      for (size_t i = 0; i < csr.col_ind().size(); ++i) {
        A.col_ind()[i] = csr.col_ind(i);
      }
      for (size_t i = 0; i < csr.val().size(); ++i) {
        A.val()[i] = csr.val(i);
      }
    }
  }

  RowPartSpmv<Ordinal, Scalar> spmv(A, 0, MPI_COMM_WORLD);

  std::shared_ptr<Start> start = std::make_shared<Start>();

  std::shared_ptr<Scatter> scatter;
  {
    Scatter::Args args{
        .dst = spmv.x_send_buf().view(), .src = spmv.lx().view(), .idx = spmv.x_send_idx().view()};
    scatter = std::make_shared<Scatter>(args);
  }

  std::shared_ptr<SpMV<Ordinal, Scalar>> yl, yr;
  {
    SpMV<Ordinal, Scalar>::Args rArgs, lArgs;
    rArgs.a = spmv.rA().view();
    rArgs.y = spmv.ly().view(); // FIXME: remote y?
    rArgs.x = spmv.rx().view();
    lArgs.a = spmv.lA().view();
    lArgs.y = spmv.ly().view();
    lArgs.x = spmv.lx().view();
    yl = std::make_shared<SpMV<Ordinal, Scalar>>("yl", lArgs);
    yr = std::make_shared<SpMV<Ordinal, Scalar>>("yr", rArgs);
  }

  std::shared_ptr<PostSend> postSend;
  std::shared_ptr<WaitSend> waitSend;
  {
    PostSend::Args args;
    for (auto &arg : spmv.send_params()) {
      if (arg.displ + arg.count > spmv.x_send_buf().size())
        throw std::logic_error(AT);
      if (!spmv.x_send_buf().data())
        throw std::logic_error(AT);
      args.sends.push_back(Isend::Args{.buf = spmv.x_send_buf().data() + arg.displ,
                                       .count = arg.count,
                                       .datatype = MPI_FLOAT,
                                       .dest = arg.dst,
                                       .tag = 0,
                                       .comm = MPI_COMM_WORLD,
                                       .request = &arg.req});
    }
    postSend = std::make_shared<PostSend>(args);
    waitSend = std::make_shared<WaitSend>(args);
  }

  std::shared_ptr<PostRecv> postRecv;
  std::shared_ptr<WaitRecv> waitRecv;
  {
    PostRecv::Args args;
    for (auto &arg : spmv.recv_params()) {
      if (arg.displ + arg.count > spmv.rx().size())
        throw std::logic_error(AT);
      if (!spmv.rx().data())
        throw std::logic_error(AT);
      args.recvs.push_back(Irecv::Args{.buf = spmv.rx().data() + arg.displ,
                                       .count = arg.count,
                                       .datatype = MPI_FLOAT,
                                       .source = arg.src,
                                       .tag = 0,
                                       .comm = MPI_COMM_WORLD,
                                       .request = &arg.req});
    }
    postRecv = std::make_shared<PostRecv>(args);
    waitRecv = std::make_shared<WaitRecv>(args);
  }
  std::shared_ptr<VectorAdd> y;
  {
    VectorAdd::Args args;
    y = std::make_shared<VectorAdd>("y", args);
  }
  std::shared_ptr<End> end = std::make_shared<End>();

  std::cerr << "create graph\n";
  Graph<OpBase> orig(start);

  // immediately recv, local spmv, or scatter
  orig.then(start, yl);
  orig.then(start, postRecv);
  orig.then(orig.then(start, scatter), postSend);

  // remote matrix after recv
  orig.then(waitRecv, yr);

  // add after local and remote done, then end
  orig.then(yl, y);
  orig.then(yr, y);

  // end once add and send is done
  orig.then(y, end);
  orig.then(waitSend, end);

  // initiate sends and recvs before waiting for either
  orig.then(postSend, waitSend);
  orig.then(postSend, waitRecv);
  orig.then(postRecv, waitSend);
  orig.then(postRecv, waitRecv);

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