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
class SpMV : public Operation
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
    cudaStream_t stream_;
    SpMV(const std::string name, Args args, cudaStream_t stream) : name_(name), args_(args), stream_(stream) {}
    std::string name() override { return name_ + "(" + std::to_string(uintptr_t(stream_)) + ")"; }

    virtual void run() override
    {
        std::cerr << "spmv: A[" << args_.a.num_rows() << "," << args_.a.num_cols() << "] * x[" << args_.x.size() << "] = y[" << args_.y.size() << "]\n";
        LAUNCH((spmv<Ordinal, Scalar>), 128, 100, 0, stream_, args_.y, args_.a, args_.x);
        CUDA_RUNTIME(cudaGetLastError());
    }
};

/* y[i] += a[i]
*/
class VectorAdd : public Operation
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
class Scatter : public Operation
{
public:
    struct Args
    {
        ArrayView<float> dst;
        ArrayView<float> src;
        ArrayView<int> idx;
    };
    Args args_;
    cudaStream_t stream_;
    Scatter(Args args, cudaStream_t stream) : args_(args), stream_(stream) {}
    std::string name() override { return "Scatter(" + std::to_string(uintptr_t(stream_)) + ")"; }

    virtual void run() override
    {
        LAUNCH(scatter, 128, 100, 0, stream_, args_.dst, args_.src, args_.idx);
        CUDA_RUNTIME(cudaGetLastError());
    }

};

class StreamSync : public Operation
{
public:
    cudaStream_t stream_;
    StreamSync(cudaStream_t stream) : stream_(stream) {}
    std::string name() override { return "StreamSync(" + std::to_string(uintptr_t(stream_)) + ")"; }
    virtual void run() override
    {
        CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    }
};

class PostRecv : public Operation
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
        std::cerr << "run PostRecv\n";
    }
};

class WaitRecv : public Operation
{
public:
    typedef PostRecv::Args Args;
    Args args_;
    WaitRecv(Args args) : args_(args) {}
    std::string name() override { return "WaitRecv"; }
};

class PostSend : public Operation
{
public:
    struct Args
    {
        std::vector<IsendArgs> sends;
    };
    Args args_;
    PostSend(Args args) : args_(args) {}
    std::string name() override { return "PostSend"; }
};

class WaitSend : public Operation
{
public:
    typedef PostSend::Args Args;
    Args args_;
    WaitSend(Args args) : args_(args) {}
    std::string name() override { return "WaitSend"; }
};