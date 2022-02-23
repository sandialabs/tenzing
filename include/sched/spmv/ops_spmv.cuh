/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include "sched/cuda/cuda_runtime.hpp"
#include "sched/operation.hpp"
#include "sched/mpi/ops_mpi.hpp"
#include "sched/cuda/ops_cuda.hpp"

#include <mpi.h>
#include <cusparse.h>

#include <iostream>
#include <vector>

#include "csr_mat.hpp"


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
class SpMV : public GpuOp
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



private:

    // arguments for cusparseSpMV
    struct CusparseArgs {
        cusparseHandle_t     handle;
        cusparseOperation_t  opA;
        Scalar               alpha;
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX;
        Scalar               beta;
        cusparseDnVecDescr_t vecY;
        cudaDataType         computeType;
        cusparseSpMVAlg_t    alg;
        void*                externalBuffer;
    };

    /* 
    create cuSPARSE arguments from SpMV arguments
    */
    CusparseArgs cusparse_from_args() {

        CusparseArgs ret{};

        CUSPARSE(cusparseCreate(&ret.handle));
        ret.opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        ret.alpha = 1;

        static_assert(std::is_same<Scalar, float>::value, "Scalar must be float");
        static_assert(std::is_same<Ordinal, int>::value, "Scalar must be float");
        cusparseIndexType_t csrRowOffsetsType = CUSPARSE_INDEX_32I;
        cusparseIndexType_t csrColIndType = CUSPARSE_INDEX_32I;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
        cudaDataType          valueType = CUDA_R_32F;
        CUSPARSE(cusparseCreateCsr(&ret.matA,
            args_.a.num_rows(), args_.a.num_cols(), args_.a.nnz(),
            args_.a.row_ptr(), args_.a.col_ind(), args_.a.val(),
            csrRowOffsetsType,
            csrColIndType,
            idxBase,
            valueType
        ));

        
        CUSPARSE(cusparseCreateDnVec(&ret.vecX, args_.x.size(), args_.x.data(), valueType));
        ret.beta = 0;
        CUSPARSE(cusparseCreateDnVec(&ret.vecY, args_.y.size(), args_.y.data(), valueType));
        ret.computeType = CUDA_R_32F;
#if CUSPARSE_VER_MAJOR >= 11
        ret.alg = CUSPARSE_SPMV_CSR_ALG2;
#else
        ret.alg = CUSPARSE_CSRMV_ALG2; // deprecated
#endif
        
        size_t bufferSize;
        CUSPARSE(cusparseSpMV_bufferSize(
            ret.handle,
            ret.opA,
            &ret.alpha,
            ret.matA,
            ret.vecX,
            &ret.beta,
            ret.vecY,
            ret.computeType,
            ret.alg,
            &bufferSize));

        CUDA_RUNTIME(cudaMalloc(&ret.externalBuffer, bufferSize));

        return ret;
    }

public:

    CusparseArgs cusparseArgs_;
    SpMV(const std::string name, Args args) : name_(name), args_(args) {
        cusparseArgs_ = cusparse_from_args();
    }

    /*
    need a new set of cuSparse arguments
    */
    SpMV(const SpMV &other) : name_(other.name_), args_(other.args_) {
        cusparseArgs_ = cusparse_from_args();
    }
    SpMV(SpMV &&other) = delete;
    ~SpMV() {
        CUSPARSE(cusparseDestroy(cusparseArgs_.handle));
        CUDA_RUNTIME(cudaFree(cusparseArgs_.externalBuffer));
        cusparseArgs_ = {};
    }

    std::string name() const override { return name_; }

    virtual void run(cudaStream_t stream) override
    {
#if 0
        // std::cerr << "spmv: A[" << args_.a.num_rows() << "," << args_.a.num_cols() << "] * x[" << args_.x.size() << "] = y[" << args_.y.size() << "]\n";
        // spmv<Ordinal, Scalar><<<128, 100, 0, stream>>>(args_.y, args_.a, args_.x);
        // CUDA_RUNTIME(cudaGetLastError());
#endif
        // std::cerr << "A.cols=" << args_.a.num_cols() << " x.size=" << args_.x.size() << "\n";
        CUSPARSE(cusparseSetStream(cusparseArgs_.handle, stream));
        CUSPARSE(
            cusparseSpMV(
                cusparseArgs_.handle,
                cusparseArgs_.opA,
                &cusparseArgs_.alpha,
                cusparseArgs_.matA,
                cusparseArgs_.vecX,
                &cusparseArgs_.beta,
                cusparseArgs_.vecY,
                cusparseArgs_.computeType,
                cusparseArgs_.alg,
                cusparseArgs_.externalBuffer
        ));
    }


    CLONE_DEF(SpMV);
    EQ_DEF(SpMV);
    LT_DEF(SpMV);

    bool operator==(const SpMV &rhs) const {
        return args_ == rhs.args_;
    }
    bool operator<(const SpMV &rhs) const {
        return name() < rhs.name();
    }
};

/* y[i] += a[i]
*/
class VectorAdd : public GpuOp
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
    std::string name() const override { return name_; }


    CLONE_DEF(VectorAdd);
    EQ_DEF(VectorAdd);
    LT_DEF(VectorAdd);
    bool operator==(const VectorAdd &rhs) const {
        return args_ == rhs.args_;
    }
    bool operator<(const VectorAdd &rhs) const {
        return name() < rhs.name();
    }
    virtual void run(cudaStream_t stream);
};

/* 
   dst[i] = src[idx[i]]
   dst[0..n]
   idx[0..n]
   src[..]
*/
class Scatter : public GpuOp
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
    std::string name() const override { return "Pack"; }

    virtual void run(cudaStream_t stream) override
    {
#ifdef SANITY_CHECKS
        if (args_.dst.size() != args_.idx.size()) {
            throw std::runtime_error(AT);
        }
#endif
        scatter<<<128, 100, 0, stream>>>(args_.dst, args_.src, args_.idx);
        CUDA_RUNTIME(cudaGetLastError());
    }


    CLONE_DEF(Scatter);
    EQ_DEF(Scatter);
    LT_DEF(Scatter);
    bool operator==(const Scatter &rhs) const {
        return args_ == rhs.args_;
    }
    bool operator<(const Scatter &rhs) const {
        return name() < rhs.name();
    }
};



class PostRecv : public CpuOp
{
public:
    struct Args
    {
        std::vector<Irecv::Args> recvs;
        bool operator==(const Args &rhs) const {
            return recvs == rhs.recvs;
        }
    };
    Args args_;
    PostRecv(Args args) : args_(args) {}
    std::string name() const override { return "PostRecv"; }
    virtual void run(Platform &/*plat*/) override
    {
        // std::cerr << "Irecvs...\n";
        for (Irecv::Args &args : args_.recvs) {
            // if (!args.buf) throw std::runtime_error(AT);
            // if (!args.request) throw std::runtime_error(AT);
            MPI_Irecv(args.buf, args.count, args.datatype, args.source, args.tag, args.comm, args.request);
        }
        // std::cerr << "Irecvs done\n";
    }


    CLONE_DEF(PostRecv);
    EQ_DEF(PostRecv);
    LT_DEF(PostRecv);
    bool operator==(const PostRecv &rhs) const {
        return args_ == rhs.args_;
    }
    bool operator<(const PostRecv &rhs) const {
        return name() < rhs.name();
    }
};

class WaitRecv : public CpuOp
{
public:
    typedef PostRecv::Args Args;
    Args args_;
    WaitRecv(Args args) : args_(args) {}
    std::string name() const override { return "WaitRecv"; }

    virtual void run(Platform &/*plat*/) override
    {
        // std::cerr << "wait(Irecvs)...\n";
        for (Irecv::Args &args : args_.recvs) {
            // if (!args.request) throw std::runtime_error(AT);
            MPI_Wait(args.request, MPI_STATUS_IGNORE);
        }
        // std::cerr << "wait(Irecvs) done\n";
    }


    CLONE_DEF(WaitRecv);
    EQ_DEF(WaitRecv);
    LT_DEF(WaitRecv);
    bool operator==(const WaitRecv &rhs) const {
        return args_ == rhs.args_;
    }
    bool operator<(const WaitRecv &rhs) const {
        return name() < rhs.name();
    }
};

class PostSend : public CpuOp
{
public:
    struct Args
    {
        std::vector<Isend::Args> sends;
        bool operator==(const Args &rhs) const {
            return sends == rhs.sends;
        }
    };
    Args args_;
    PostSend(Args args) : args_(args) {}
    std::string name() const override { return "PostSend"; }

    virtual void run(Platform &/*plat*/) override
    {
        // std::cerr << "Isends...\n";
        for (Isend::Args &args : args_.sends) {
            // if (!args.buf) throw std::runtime_error(AT);
            // if (!args.request) throw std::runtime_error(AT);
            MPI_Isend(args.buf, args.count, args.datatype, args.dest, args.tag, args.comm, args.request);
        }
        // std::cerr << "Isends done\n";
    }


    CLONE_DEF(PostSend);
    EQ_DEF(PostSend);
    LT_DEF(PostSend);
    bool operator==(const PostSend &rhs) const {
        return args_ == rhs.args_;
    }
    bool operator<(const PostSend &rhs) const {
        return name() < rhs.name();
    }
};

class WaitSend : public CpuOp
{
public:
    typedef PostSend::Args Args;
    Args args_;
    WaitSend(Args args) : args_(args) {}
    std::string name() const override { return "WaitSend"; }

    virtual void run(Platform &/*plat*/) override
    {
        // std::cerr << "wait(Isends)...\n";
        for (Isend::Args &args : args_.sends) {
            // if (!args.request) throw std::runtime_error(AT);
            MPI_Wait(args.request, MPI_STATUS_IGNORE);
        }
        // std::cerr << "wait(Isends) done\n";
    }


    CLONE_DEF(WaitSend);
    EQ_DEF(WaitSend);
    LT_DEF(WaitSend);
    bool operator==(const WaitSend &rhs) const {
        return args_ == rhs.args_;
    }
    bool operator<(const WaitSend &rhs) const {
        return name() < rhs.name();
    }
};


