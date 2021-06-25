#pragma once

#include "operation.hpp"

#include "fake_cuda.hpp"
#include "fake_mpi.hpp"

#include <vector>

class SpMV : public Operation
{
public:
    struct Args {
        float *y;
        int yN;
        float *x;
        int xN;
        int *rowPtr;
        int *colInd;
        float *colVal;
        int nRow;
    };

    std::string name_;
    Args args_;
    cudaStream_t stream_;
    SpMV(const std::string name, Args args, cudaStream_t stream) : name_(name), args_(args), stream_(stream) {}
    std::string name() override { return name_ + "(" + std::to_string(uintptr_t(stream_)) + ")"; }
};

/* y[i] += a[i]
*/
class VectorAdd : public Operation
{
public:
struct Args {
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
   dst[0..dstN]
   src[0..srcN]
   idx[0..srcN]
*/
class Scatter : public Operation
{
public:
    struct Args {
        float *dst;
        int dstN;
        float *src;
        int srcN;
        int *idx;
    };
    Args args_;
    cudaStream_t stream_;
    Scatter(Args args, cudaStream_t stream) : args_(args), stream_(stream) {}
    std::string name() override { return "Scatter(" + std::to_string(uintptr_t(stream_)) + ")"; }
};

class StreamSync : public Operation
{
public:
    cudaStream_t stream_;
    StreamSync(cudaStream_t stream) : stream_(stream) {}
    std::string name() override { return "StreamSync(" + std::to_string(uintptr_t(stream_)) + ")"; }
};

class PostRecv : public Operation
{
public:
    struct Args {
        std::vector<IrecvArgs> recvs; 
    };
    Args args_;
    PostRecv(Args args) : args_(args) {}
    std::string name() override { return "PostRecv"; }
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
    struct Args {
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