/* use MCTS on a particular assignment of operations to streams

   yl is in stream1, other GPU operations in stream2
*/

#include "sched/cuda_runtime.h"
#include "sched/schedule.hpp"
#include "sched/graph.hpp"
#include "sched/numeric.hpp"
#include "sched/numa.hpp"
#include "sched/benchmarker.hpp"

#include "ops_spmv.cuh"

#include "where.hpp"
#include "csr_mat.hpp"
#include "row_part_spmv.cuh"

#include "mm/mm.hpp"

#include <algorithm>
#include <numeric>
#include <chrono>
#include <thread>
#include <iostream>

#include <cusparse.h>

typedef int Ordinal;
typedef float Scalar;
typedef MtxReader<Ordinal, Scalar> reader_t;
typedef typename reader_t::coo_type mm_coo_t;
typedef typename reader_t::csr_type mm_csr_t;

template <Where w>
using csr_type = CsrMat<w, Ordinal, Scalar>;

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    bind_to_local_memory();

    MPI_Barrier(MPI_COMM_WORLD);

    {

        char hostname[MPI_MAX_PROCESSOR_NAME] = {};
        int len;
        MPI_Get_processor_name(hostname, &len);

        const char *p = std::getenv("OMP_PLACES");
        if (!p) p = "<unset>";
        std::cerr << "rank " << rank << " on " << hostname << " OMP_PLACES: " << p << "\n";

        // round-robin GPU scheduling
        int devcount;
        CUDA_RUNTIME(cudaGetDeviceCount(&devcount));
        int dev = rank % devcount;
        CUDA_RUNTIME(cudaSetDevice(dev));

        cudaDeviceProp prop;
        CUDA_RUNTIME(cudaGetDeviceProperties(&prop, dev));
        fprintf(stderr, "rank %d on %s GPU %08x:%02x:%02x.0 (%d)\n", rank, hostname, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, dev);

    }

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    /* ensure streams are numerically ordered, so that later when ranks sort by stream,
       stream1 is the smallest on both ranks
    */
    if (stream1 > stream2) std::swap(stream1, stream2);



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
        if (0 == rank)
        {
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

    std::shared_ptr<StreamedOp> scatter;
    {
        Scatter::Args args{
            .dst = spmv.x_send_buf().view(),
            .src = spmv.lx().view(),
            .idx = spmv.x_send_idx().view()};
        auto _scatter = std::make_shared<Scatter>(args);
        scatter = std::make_shared<StreamedOp>(_scatter, stream2);
    }

    std::shared_ptr<StreamedOp> yl, yr;
    {
        SpMV<Ordinal, Scalar>::Args rArgs, lArgs;
        rArgs.a = spmv.rA().view();
        rArgs.y = spmv.ly().view(); // FIXME: remote y?
        rArgs.x = spmv.rx().view();
        lArgs.a = spmv.lA().view();
        lArgs.y = spmv.ly().view();
        lArgs.x = spmv.lx().view();
        auto _yl = std::make_shared<SpMV<Ordinal, Scalar>>("yl", lArgs);
        auto _yr = std::make_shared<SpMV<Ordinal, Scalar>>("yr", rArgs);

        // yl and yr in different streams
        yl = std::make_shared<StreamedOp>(_yl, stream1);
        yr = std::make_shared<StreamedOp>(_yr, stream2);
    }

    std::shared_ptr<PostSend> postSend;
    std::shared_ptr<WaitSend> waitSend;
    {
        PostSend::Args args;
        for (auto &arg : spmv.send_params())
        {
            if (arg.displ + arg.count > spmv.x_send_buf().size()) throw std::logic_error(AT);
            if (!spmv.x_send_buf().data()) throw std::logic_error(AT);
            args.sends.push_back(Isend::Args{
                .buf = spmv.x_send_buf().data() + arg.displ,
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
        for (auto &arg : spmv.recv_params())
        {
            if (arg.displ + arg.count > spmv.rx().size()) throw std::logic_error(AT);
            if (!spmv.rx().data()) throw std::logic_error(AT);
            args.recvs.push_back(Irecv::Args{
                .buf = spmv.rx().data() + arg.displ,
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
    std::shared_ptr<StreamedOp> y;
    {
        VectorAdd::Args args;
        auto _y = std::make_shared<VectorAdd>("y", args);
        y = std::make_shared<StreamedOp>(_y, stream2);
    }
    std::shared_ptr<End> end = std::make_shared<End>();

    std::cerr << "create graph\n";
    Graph<Node> orig(start);

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

    std::vector<Graph<Node>> gpuGraphs;
    gpuGraphs.push_back(orig);

    if (0 == rank) {
        std::cerr << gpuGraphs.size() << " GpuNode graphs\n";
    }

    if (0 == rank) {
#if 1
        for (auto &graph : gpuGraphs) {
            graph.dump();
            std::cerr << "\n";
        }
#endif
#if 0
        gpuGraphs.begin()->dump();
        std::cerr << "\n";
        (--gpuGraphs.end())->dump();
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);

    
    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "insert sync...\n";
    std::vector<Graph<Node>> syncedGraphs;
    for (auto &graph : gpuGraphs) {
        auto next = insert_synchronization(graph);
        syncedGraphs.push_back(next);
    }

    if (0 == rank) {
        std::cerr << "created " << syncedGraphs.size() << " sync graphs:\n";
    }


    if (0 == rank) {
        syncedGraphs.begin()->dump();
        std::cerr << "\n";
    }



    if (0 == rank) {
        for (auto &graph : syncedGraphs) {
            graph.dump();
            std::cerr << "\n";
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "convert to cpu graphs...\n";
    std::vector<Graph<CpuNode>> cpuGraphs;
    for (auto &graph : syncedGraphs) {
        cpuGraphs.push_back(graph.nodes_cast<CpuNode>());
    }
    if (0 == rank) std::cerr << "converted " << cpuGraphs.size() << " graphs\n";

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "create orderings...\n";
    std::vector<Schedule> schedules;
    for (auto &graph : cpuGraphs) {
        auto ss = make_schedules(graph);
        for (auto &s : ss) {
            schedules.push_back(s);
        }
    }
    std::cerr << "created " << schedules.size() << " schedules\n";


    
    /* Many places, the order of traversal is specified by a pointer address, which is different in different address spaces
    This means that some kind of canonical order must be imposed on the generated schedules that is the same for each rank

    schedules with stream swaps are considered equal, but not all pairs of streams compare equally.
    so, we need to sort the schedules to ensure the same duplicates are deleted on all ranks
    */
    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "sort schedules...\n";
    std::sort(schedules.begin(), schedules.end(), Schedule::by_node_typeid);


    std::vector<Schedule> sas, sbs;
    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "eliminate equivalent schedules...\n";
    {
        int count = 0;
        size_t total = schedules.size() * (schedules.size() - 1);
        int next = 99;
        for (size_t i = 0; i < schedules.size(); ++i) {
            for (size_t j = i+1; j < schedules.size(); ++j) {
                if (Schedule::predicate(schedules[i], schedules[j])) {
                    sas.push_back(schedules[i]);
                    sbs.push_back(schedules[j]);
                    schedules.erase(schedules.begin() + j);
                    size_t left = (schedules.size() - i) * (schedules.size() - i - 1);
                    if (left < next * total / 100) {
                        if (0 == rank) std::cerr << next << "% (~" << (schedules.size()-i) * (schedules.size() - i - 1) << " comparisons left...)\n";
                        next = left * 100 / total;
                    }
                    
                    count += 1;
                    --j; // since we need to check the schedule that is now in j
                }
            }
        }
        std::cerr << "found " << count << " duplicate schedules\n";
        std::cerr << "found " << schedules.size() << " unique schedules\n";
    }


    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (0 == rank)
    {
        for (size_t i = 0; i < schedules.size(); ++i) {
            
            std::cerr << i;
            for (std::shared_ptr<CpuNode> op : schedules[i].order)
            {
                std::cerr <<  "\t" << op->json();
            }
            std::cerr << "\n";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for(std::chrono::seconds(1));


    // test all schedules

    if (0 == rank) std::cerr << "testing schedules...\n";
    for (size_t i = 0; i < schedules.size(); ++i) {
    // for (size_t i = 9; i < 10; ++i) {
        if (0 == rank) std::cerr << " " << i;
        MPI_Barrier(MPI_COMM_WORLD);
        schedules[i].run();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (0 == rank) std::cerr << std::endl;
    if (0 == rank) std::cerr << "done" << std::endl;

    // measured times for each schedule
    std::vector<Benchmark::Result> times;

    BenchOpts opts;
    opts.nIters = 50;
    EmpiricalBenchmarker benchmarker;


    if (0 == rank) std::cout << "1pctl,10pctl,50pctl,90pct,99pct,stddev,order\n";
    for (auto &schedule : schedules) {

        Benchmark::Result br = benchmarker.benchmark(schedule.order, MPI_COMM_WORLD, opts);

        if (0 == rank) {
            std::cout << br.pct01
            << "," << br.pct10
            << "," << br.pct50
            << "," << br.pct90
            << "," << br.pct99
            << "," << br.stddev << "\n"
            << std::flush;
        }
    }

    MPI_Finalize();

}