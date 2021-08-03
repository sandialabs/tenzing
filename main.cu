#include <iostream>

// wrappers
#include "mpi.h"
#include "cuda_runtime.h"

#include "ops_spmv.cuh"
#include "schedule.hpp"
#include "graph.hpp"

#include "where.hpp"
#include "csr_mat.hpp"
#include "row_part_spmv.cuh"
#include "numeric.hpp"

#include <algorithm>
#include <numeric>
#include <chrono>
#include <thread>

typedef int Ordinal;
typedef float Scalar;

template <Where w>
using csr_type = CsrMat<w, Ordinal, Scalar>;
using coo_type = CooMat<Ordinal, Scalar>;

int main(int argc, char **argv)
{

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

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    /* ensure streams are numerically ordered, so that later when ranks sort by stream,
       stream1 is the smallest on both ranks
    */
    if (stream1 > stream2) std::swap(stream1, stream2);

    int m = 1500000;
    int n = m;
    int bw = m / size;
    int nnz = m * 10;

    csr_type<Where::host> A;
    // generate and distribute A
    if (0 == rank)
    {
        std::cerr << "generate matrix\n";
        A = random_band_matrix<Ordinal, Scalar>(m, bw, nnz);
    }

    RowPartSpmv<Ordinal, Scalar> spmv(A, 0, MPI_COMM_WORLD);

    std::shared_ptr<Start> start = std::make_shared<Start>();

    std::shared_ptr<Scatter> scatter;
    {
        Scatter::Args args{
            .dst = spmv.x_send_buf().view(),
            .src = spmv.lx().view(),
            .idx = spmv.x_send_idx().view()};
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
        for (auto &arg : spmv.send_params())
        {
            if (arg.displ + arg.count > spmv.x_send_buf().size()) throw std::logic_error(AT);
            if (!spmv.x_send_buf().data()) throw std::logic_error(AT);
            args.sends.push_back(IsendArgs{
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
            args.recvs.push_back(IrecvArgs{
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
    std::shared_ptr<VectorAdd> y;
    {
        VectorAdd::Args args;
        y = std::make_shared<VectorAdd>("y", args);
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
    if (0 == rank) {
        std::cerr << "apply streams\n";
    }
    std::vector<Graph<Node>> gpuGraphs = use_streams(orig, {stream1, stream2});

    if (0 == rank) {
        std::cerr << "created " << gpuGraphs.size() << " GpuNode graphs\n";
    }

    if (0 == rank) {
        for (auto &graph : gpuGraphs) {
            graph.dump();
            std::cerr << "\n";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "insert sync...\n";
    std::vector<Graph<Node>> syncedGraphs;
    for (auto &graph : gpuGraphs) {
        auto next = insert_synchronization(graph);
        syncedGraphs.push_back(next);
    }
    if (0 == rank) {
        std::cerr << "created " << syncedGraphs.size() << " sync graphs:\n";
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


    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) std::cerr << "eliminate equivalent schedules...\n";
    {
        int count = 0;
        size_t total = schedules.size() * (schedules.size() - 1);
        int next = 99;
        for (size_t i = 0; i < schedules.size(); ++i) {
            for (size_t j = i+1; j < schedules.size(); ++j) {
                if (Schedule::predicate(schedules[i], schedules[j])) {
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
                std::cerr <<  ", " << op->name();
            }
            std::cerr << "\n";
        }
    }

#if 0
    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    if (1 == rank)
    {
        for (size_t i = 0; i < schedules.size(); ++i) {
            
            std::cerr << i;
            for (std::shared_ptr<CpuNode> op : schedules[i].order)
            {
                std::cerr <<  ", " << op->name();
            }
            std::cerr << "\n";
        }
    }
#endif

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


    // order to run schedules in each iteration
    std::vector<int> perm(schedules.size());
    std::iota(perm.begin(), perm.end(), 0);
    // measured times for each schedule
    std::vector<std::vector<double>> times(schedules.size());

    // each iteration, do schedules in a random order
    for (int i = 0; i < 1000; ++i) {
        if (0 == rank) {
            std::cerr << " " << i;
        }
        if (0 == rank) {
            std::random_shuffle(perm.begin(), perm.end());
        }
        MPI_Bcast(perm.data(), perm.size(), MPI_INT, 0, MPI_COMM_WORLD);
        for (int si : perm)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            double start = MPI_Wtime();
            schedules[si].run();
            double elapsed = MPI_Wtime() - start;
            times[si].push_back(elapsed);
        }
    }
    if (0 == rank) {
        std::cerr << std::endl;
    }

    // for each schedule
    for (size_t i = 0; i < times.size(); ++i)
    {
        // the iteration time is the maximum observed across all ranks
        MPI_Allreduce(MPI_IN_PLACE, times[i].data(), times[i].size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    MPI_Finalize();

#if 0
    // https://stackoverflow.com/questions/17074324/how-can-i-sort-two-vectors-in-the-same-way-with-criteria-that-uses-only-one-of
    std::vector<int> perm(times.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](size_t i, size_t j)
              { return times[i][times[i].size() / 2] < times[j][times[j].size() / 2]; })
#endif


    if (0 == rank)
    {
        std::cout << "1pctl,10pctl,50pctl,90pct,99pct,stddev,order\n";
        for (auto &st : times)
        {
            std::sort(st.begin(), st.end());
            std::cout << st[st.size() / 100] 
                      << "," << st[st.size() / 10] 
                      << "," << st[st.size() / 2] 
                      << "," << st[st.size() * 9 / 10] 
                      << "," << st[st.size() * 99 / 100] 
                      << "," << stddev(st) << "\n";
        }


        // features of each result
        // [0] = yl & yr in the same stream
        // [1] = yl before post send
        std::vector<std::array<int, 2>> features(times.size(), {0});
        for (size_t i = 0; i < times.size(); ++i) {
            // feature 0
            cudaStream_t s = nullptr;
            for (auto p : schedules[i].order) {
                if (auto spmv = std::dynamic_pointer_cast<SpMV<Ordinal, Scalar>>(p)) {
                    if (!s) {
                        s = spmv->stream();
                    } else if (s == spmv->stream()) {
                        features[i][0] = 1;
                        break;
                    } else {
                        features[i][0] = 0;
                        break;
                    }
                }
            }

            // feature 1
            for (auto &p : schedules[i].order) {
                if (std::dynamic_pointer_cast<SpMV>(p)) {
                    features[i][1] = 1;
                    break; // yl came first
                }
                if (std::dynamic_pointer_cast<PostSend>(p)) {
                    features[i][1] = 0;
                    break; // PostSend came first
                }
            }
        }

        for (size_t i = 0; i < features.size(); ++i) {
            std::cout << i 
                    << "," << features[i][0]
                    << "," << features[i][1]
                    << "\n";
        }
    }

}