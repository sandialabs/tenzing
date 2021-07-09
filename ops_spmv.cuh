#pragma once

// wrappers
#include "cuda_runtime.h"
#include "mpi.h"

#include "operation.hpp"

#include "csr_mat.hpp"

#include <iostream>
#include <vector>

#define __CLASS__ std::remove_reference<decltype(classMacroImpl(this))>::type
template<class T> T& classMacroImpl(const T* t);

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
        Ordinal j = idx(i);
        dst(i) = src(j);
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
        bool operator==(const Args &rhs) const {
            return a == rhs.a && x == rhs.x && y == rhs.y;
        }
    };

    std::string name_;
    Args args_;
    SpMV(const std::string name, Args args) : name_(name), args_(args) {}
    std::string name() override { return name_; }
    EQUAL_DEF_1(SpMV)
    {
        return args_ == p->args_;
    }
    EQUAL_DEF_2

    virtual void run(cudaStream_t stream) override
    {
        // std::cerr << "spmv: A[" << args_.a.num_rows() << "," << args_.a.num_cols() << "] * x[" << args_.x.size() << "] = y[" << args_.y.size() << "]\n";
        LAUNCH((spmv<Ordinal, Scalar>), 128, 100, 0, stream, args_.y, args_.a, args_.x);
        // spmv<Ordinal, Scalar><<<128, 100, 0, stream>>>(args_.y, args_.a, args_.x);
        CUDA_RUNTIME(cudaGetLastError());
    }

    virtual std::unique_ptr<Node> clone() override {return std::unique_ptr<Node>(static_cast<Node*>(new SpMV<Ordinal, Scalar>(*this)));}
    virtual int tag() const override { return 4; }
};

/* y[i] += a[i]
*/
class VectorAdd : public GpuNode
{
public:
    struct Args
    {
        float *y;
        float *a;
        int n;
        bool operator==(const Args &rhs) const {
            return y == rhs.y && a == rhs.a && n == rhs.n;
        }
    };
    std::string name_;
    Args args_;
    VectorAdd(const std::string name, Args args) : name_(name), args_(args) {}
    std::string name() override { return name_; }
    EQUAL_DEF_1(VectorAdd)
    {
        return args_ == p->args_;
    }
    EQUAL_DEF_2

    virtual std::unique_ptr<Node> clone() override {return std::unique_ptr<Node>(static_cast<Node*>(new __CLASS__(*this)));}
    virtual int tag() const override { return 5; }
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
        bool operator==(const Args &rhs) const {
            return dst == rhs.dst && src == rhs.src && idx == rhs.idx;
        }
    };
    Args args_;
    Scatter(Args args) : args_(args) {}
    std::string name() override { return "Scatter"; }
    EQUAL_DEF_1(Scatter)
    {
        return args_ == p->args_;
    }
    EQUAL_DEF_2

    virtual void run(cudaStream_t stream) override
    {
#ifdef SANITY_CHECKS
        if (args_.dst.size() != args_.idx.size()) {
            throw std::runtime_error(AT);
        }
#endif
        LAUNCH(scatter, 128, 100, 0, stream, args_.dst, args_.src, args_.idx);
        CUDA_RUNTIME(cudaGetLastError());
    }

    virtual std::unique_ptr<Node> clone() override {return std::unique_ptr<Node>(static_cast<Node*>(new __CLASS__(*this)));}
    virtual int tag() const override { return 6; }

};



class PostRecv : public CpuNode
{
public:
    struct Args
    {
        std::vector<IrecvArgs> recvs;
        bool operator==(const Args &rhs) const {
            return recvs == rhs.recvs;
        }
    };
    Args args_;
    PostRecv(Args args) : args_(args) {}
    std::string name() override { return "PostRecv"; }
    EQUAL_DEF_1(PostRecv)
    {
        return args_ == p->args_;
    }
    EQUAL_DEF_2
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

    virtual std::unique_ptr<Node> clone() override {return std::unique_ptr<Node>(static_cast<Node*>(new __CLASS__(*this)));}
    virtual int tag() const override { return 7; }
};

class WaitRecv : public CpuNode
{
public:
    typedef PostRecv::Args Args;
    Args args_;
    WaitRecv(Args args) : args_(args) {}
    std::string name() override { return "WaitRecv"; }
    EQUAL_DEF_1(WaitRecv)
    {
        return args_ == p->args_;
    }
    EQUAL_DEF_2
    virtual void run() override
    {
        // std::cerr << "wait(Irecvs)...\n";
        for (IrecvArgs &args : args_.recvs) {
            // if (!args.request) throw std::runtime_error(AT);
            MPI_Wait(args.request, MPI_STATUS_IGNORE);
        }
        // std::cerr << "wait(Irecvs) done\n";
    }

    virtual std::unique_ptr<Node> clone() override {return std::unique_ptr<Node>(static_cast<Node*>(new __CLASS__(*this)));}
    virtual int tag() const override { return 8; }
};

class PostSend : public CpuNode
{
public:
    struct Args
    {
        std::vector<IsendArgs> sends;
        bool operator==(const Args &rhs) const {
            return sends == rhs.sends;
        }
    };
    Args args_;
    PostSend(Args args) : args_(args) {}
    std::string name() override { return "PostSend"; }
    EQUAL_DEF_1(PostSend)
    {
        return args_ == p->args_;
    }
    EQUAL_DEF_2
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

    virtual std::unique_ptr<Node> clone() override {return std::unique_ptr<Node>(static_cast<Node*>(new __CLASS__(*this)));}
    virtual int tag() const override { return 9; }
};

class WaitSend : public CpuNode
{
public:
    typedef PostSend::Args Args;
    Args args_;
    WaitSend(Args args) : args_(args) {}
    std::string name() override { return "WaitSend"; }
    EQUAL_DEF_1(WaitSend)
    {
        return args_ == p->args_;
    }
    EQUAL_DEF_2
    virtual void run() override
    {
        // std::cerr << "wait(Isends)...\n";
        for (IsendArgs &args : args_.sends) {
            // if (!args.request) throw std::runtime_error(AT);
            MPI_Wait(args.request, MPI_STATUS_IGNORE);
        }
        // std::cerr << "wait(Isends) done\n";
    }

    virtual std::unique_ptr<Node> clone() override {return std::unique_ptr<Node>(static_cast<Node*>(new __CLASS__(*this)));}
    virtual int tag() const override { return 10; }
};



#undef __CLASS__