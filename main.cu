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


    // round-robin GPU scheduling
    int count;
    CUDA_RUNTIME(cudaGetDeviceCount(&count));
    int dev = rank % count;
    std::cerr << rank << " on GPU " << dev << std::endl;
    CUDA_RUNTIME(cudaSetDevice(dev));

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int m = 150000;
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
        y = std::make_shared<VectorAdd>("y", args, stream2);
    }
    std::shared_ptr<End> end = std::make_shared<End>();

    // immediately recv, local spmv, or scatter
    Node::then(start, yl);
    Node::then(start, postRecv);
    Node::then(Node::then(start, scatter), postSend);

    // remote matrix after recv
    Node::then(Node::then(waitRecv, yr), end);

    // add after local and remote done, then end
    Node::then(yl, y);
    Node::then(yr, y);

    // end once add and send is done
    Node::then(y, end);
    Node::then(waitSend, end);

    // initiate sends and recvs before waiting for either
    Node::then(postSend, waitSend);
    Node::then(postSend, waitRecv);
    Node::then(postRecv, waitSend);
    Node::then(postRecv, waitRecv);


    Graph<Node> orig(start);
    std::vector<Graph<Node>> GpuGraphs = use_streams(orig, {stream1});


#if 0
    std::vector<Schedule> schedules = make_schedules(start);

    if (0 == rank)
    {
        std::cerr << schedules.size() << " schedules:\n";
        for (size_t i = 0; i < schedules.size(); ++i) {
            
            for (CpuNode *op : schedules[i].order)
            {
                std::cerr << op->name() << ", ";
            }
            std::cerr << i;
            std::cerr << "\n";
        }
    }

    // test all schedules
    if (0 == rank)
    {
        std::cerr << "test\n";
    }
    for (size_t i = 0; i < schedules.size(); ++i) {
    // for (size_t i = 9; i < 10; ++i) {
        if (0 == rank) {
            std::cerr << " " << i;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        schedules[i].run();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    std::cerr << std::endl;

    // order to run schedules in each iteration
    std::vector<int> perm(schedules.size());
    std::iota(perm.begin(), perm.end(), 0);

    // measured times for each schedule
    std::vector<std::vector<double>> times(schedules.size());
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

    // time is maximum observed across all ranks
    for (size_t i = 0; i < times.size(); ++i)
    {
        for (size_t j = 0; j < times[i].size(); ++j)
        {
            MPI_Allreduce(MPI_IN_PLACE, times[i].data(), times[i].size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        }
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
    }

#endif
}