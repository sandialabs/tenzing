/* use MCTS for both order and streams
 */

#include <cwpearson/argparse.hpp>

#include "sched/benchmarker.hpp"
#include "sched/cuda_runtime.h"
#include "sched/graph.hpp"
#include "sched/mcts.hpp"
#include "sched/numeric.hpp"
#include "sched/platform.hpp"
#include "sched/schedule.hpp"

#include "ops_spmv.cuh"

#include "csr_mat.hpp"
#include "row_part_spmv.cuh"
#include "where.hpp"

typedef int Ordinal;
typedef float Scalar;

template <Where w> using csr_type = CsrMat<w, Ordinal, Scalar>;

template <typename Strategy> int platform_mcts(mcts::Opts &opts, int argc, char **argv) {

  opts.nIters = 300;
  opts.benchOpts.nIters = 50;
  int m = 150000; // matrix size

  bool noExpandRollout = false;
  argparse::Parser parser("SpMV design-space exporation using monte-carlo tree search");
  parser.add_option(opts.nIters, "--mcts-iters", "-i")->help("how many MCTS iterations to do");
  parser.add_option(opts.benchOpts.nIters, "--benchmark-iters", "-b")->help("how many benchmark measurements to do.");
  parser.add_option(m, "--matrix-m", "-m")->help("random matrix dimension");
  parser.add_flag(noExpandRollout, "--no-expand-rollout")->help("don't expand rollout");
  parser.no_unrecognized();

  if (!parser.parse(argc, argv)) {
    std::cerr << parser.help();
    exit(EXIT_FAILURE);
  }
  opts.expandRollout = !noExpandRollout;

  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

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


  int bw = m / size;
  int nnz = m * 10;

  csr_type<Where::host> A;

  // generate and distribute A
  if (0 == rank) {
    std::cerr << "generate matrix\n";
    A = random_band_matrix<Ordinal, Scalar>(m, bw, nnz);
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
  Platform platform = Platform::make_n_streams(2, MPI_COMM_WORLD);

  STDERR("mcts...");

  mcts::Result result = mcts::mcts<Strategy>(orig, platform, benchmarker, opts);
  if (0 == rank) result.dump_csv();

  return 0;
}