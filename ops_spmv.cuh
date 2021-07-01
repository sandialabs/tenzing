#pragma once

// wrappers
#include "cuda_runtime.h"
#include "mpi.h"

#include "operation.hpp"

#include "csr_mat.hpp"

#include <iostream>
#include <vector>

/* Ax=y
*/
template <typename Ordinal, typename Scalar>
__global__ void spmv(
    ArrayView<Scalar> y,
    const typename CsrMat<Where::device, Ordinal, Scalar>::View A,
    const ArrayView<Scalar> x)
{
    // one thread per row
    for (int r = blockDim.x * blockIdx.x + threadIdx.x; r < A.num_rows(); r += blockDim.x * gridDim.x)
    {
        Scalar acc = 0;
        for (int ci = A.row_ptr(r); ci < A.row_ptr(r + 1); ++ci)
        {
            int c = A.col_ind(ci);
            acc += A.val(ci) * x(c);
        }
        y(r) += acc;
    }
}

template <typename Ordinal, typename Scalar>
__global__ void scatter(
    ArrayView<Scalar> dst,
    const ArrayView<Scalar> src,
    const ArrayView<Ordinal> idx)
{
    // one thread per row
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < dst.size(); i += blockDim.x * gridDim.x)
    {
        dst(i) = src(idx(i));
    }
}

template<typename Ordinal, typename Scalar>
class SpMV : public GpuNode
{

public:
    struct Args
    {
        typename CsrMat<Where::device, Ordinal, Scalar>::View a;
        ArrayView<Scalar> x;
        ArrayView<Scalar> y;
    };

    std::string name_;
    Args args_;
    SpMV(const std::string name, Args args) : name_(name), args_(args) {}
    std::string name() override { return name_; }

    virtual void run(cudaStream_t stream) override
    {
        // std::cerr << "spmv: A[" << args_.a.num_rows() << "," << args_.a.num_cols() << "] * x[" << args_.x.size() << "] = y[" << args_.y.size() << "]\n";
        LAUNCH((spmv<Ordinal, Scalar>), 128, 100, 0, stream, args_.y, args_.a, args_.x);
        CUDA_RUNTIME(cudaGetLastError());
    }

};

/* y[i] += a[i]
*/
class VectorAdd : public CpuNode
{
public:
    struct Args
    {
        float *y;
        float *a;
        int n;
    };
    std::string name_;
    Args args_;
    cudaStream_t stream_;
    VectorAdd(const std::string name, Args args, cudaStream_t stream) : name_(name), args_(args), stream_(stream) {}
    std::string name() override { return name_ + "(" + std::to_string(uintptr_t(stream_)) + ")"; }
};

/* 
   dst[i] = src[idx[i]]
   dst[0..n]
   idx[0..n]
   src[..]
*/
class Scatter : public GpuNode
{
public:
    struct Args
    {
        ArrayView<float> dst;
        ArrayView<float> src;
        ArrayView<int> idx;
    };
    Args args_;
    Scatter(Args args) : args_(args) {}
    std::string name() override { return "Scatter"; }

    virtual void run(cudaStream_t stream) override
    {
        LAUNCH(scatter, 128, 100, 0, stream, args_.dst, args_.src, args_.idx);
        CUDA_RUNTIME(cudaGetLastError());
    }

};

class StreamSync : public GpuNode
{
public:
    std::string name() override { return "StreamSync"; }
    virtual void run(cudaStream_t stream) override
    {
        CUDA_RUNTIME(cudaStreamSynchronize(stream));
    }
};

/* cause waiter to wait on current state of waitee
   this node can be inserted by the scheduler when GPU operations
   in different streams are ordered

   TODO: could decouple these calls in the future?
*/
class StreamWait : public CpuNode{
    cudaEvent_t event_;
    cudaStream_t waitee_, waiter_;
    StreamWait(cudaStream_t waitee, cudaStream_t waiter) : waitee_(waitee), waiter_(waiter) {
        CUDA_RUNTIME(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    }
    ~StreamWait() {/* FIXME: stream cleanup */ }


    virtual void run() override {
        CUDA_RUNTIME(cudaEventRecord(event_, waitee_));
        CUDA_RUNTIME(cudaStreamWaitEvent(waiter_, event_, 0 /*flags*/));
    }
};

class PostRecv : public CpuNode
{
public:
    struct Args
    {
        std::vector<IrecvArgs> recvs;
    };
    Args args_;
    PostRecv(Args args) : args_(args) {}
    std::string name() override { return "PostRecv"; }
    virtual void run() override
    {
        // std::cerr << "Irecvs...\n";
        for (IrecvArgs &args : args_.recvs) {
            // if (!args.buf) throw std::runtime_error(AT);
            // if (!args.request) throw std::runtime_error(AT);
            MPI_Irecv(args.buf, args.count, args.datatype, args.source, args.tag, args.comm, args.request);
        }
        // std::cerr << "Irecvs done\n";
    }
};

class WaitRecv : public CpuNode
{
public:
    typedef PostRecv::Args Args;
    Args args_;
    WaitRecv(Args args) : args_(args) {}
    std::string name() override { return "WaitRecv"; }
    virtual void run() override
    {
        // std::cerr << "wait(Irecvs)...\n";
        for (IrecvArgs &args : args_.recvs) {
            // if (!args.request) throw std::runtime_error(AT);
            MPI_Wait(args.request, MPI_STATUS_IGNORE);
        }
        // std::cerr << "wait(Irecvs) done\n";
    }
};

class PostSend : public CpuNode
{
public:
    struct Args
    {
        std::vector<IsendArgs> sends;
    };
    Args args_;
    PostSend(Args args) : args_(args) {}
    std::string name() override { return "PostSend"; }
    virtual void run() override
    {
        // std::cerr << "Isends...\n";
        for (IsendArgs &args : args_.sends) {
            // if (!args.buf) throw std::runtime_error(AT);
            // if (!args.request) throw std::runtime_error(AT);
            MPI_Isend(args.buf, args.count, args.datatype, args.dest, args.tag, args.comm, args.request);
        }
        // std::cerr << "Isends done\n";
    }
};

class WaitSend : public CpuNode
{
public:
    typedef PostSend::Args Args;
    Args args_;
    WaitSend(Args args) : args_(args) {}
    std::string name() override { return "WaitSend"; }
    virtual void run() override
    {
        // std::cerr << "wait(Isends)...\n";
        for (IsendArgs &args : args_.sends) {
            // if (!args.request) throw std::runtime_error(AT);
            MPI_Wait(args.request, MPI_STATUS_IGNORE);
        }
        // std::cerr << "wait(Isends) done\n";
    }
};

