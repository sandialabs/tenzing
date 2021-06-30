/* Define some things that might help a C++ compiler
   compile (non-functional) CUDA code
*/

#pragma once

#include <cstdio>
#include <cstdlib>

#ifdef REAL_CUDA
#include <cuda_runtime.h>
#define LAUNCH(name, gd, bd, sm, st, ...) name<<<gd, bd, sm, st>>>(__VA_ARGS__)
#else
#define LAUNCH(name, gd, bd, sm, st, ...) name(__VA_ARGS__)

#define __host__   // no-op
#define __device__ // no-op
#define __global__ // no-op

struct dim3
{
    unsigned x;
    unsigned y;
    unsigned z;

    dim3() = default;
    dim3(unsigned _x) : x(_x), y(1), z(1) {}
    dim3(unsigned _x, unsigned _y, unsigned _z) : x(_x), y(_y), z(_z) {}
};
dim3 threadIdx(0, 0, 0);
dim3 blockDim(1, 1, 1);
dim3 blockIdx(0, 0, 0);
dim3 gridDim(1, 1, 1);

typedef int cudaStream_t;
enum cudaError_t
{
    cudaSuccess
};
enum cudaMemcpyKind
{
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost
};
enum cudaHostAllocFlags
{
    cudaHostAllocDefault
};

cudaError_t cudaStreamCreate(cudaStream_t *stream)
{
    static int num = 0;
    *stream = (++num);
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t)
{
    return cudaSuccess;
}
cudaError_t cudaStreamSynchronize(cudaStream_t)
{
    return cudaSuccess;
}

const char *cudaGetErrorString(cudaError_t)
{
    return "cudaSuccess";
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t n, cudaMemcpyKind, cudaStream_t)
{
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
cudaError_t cudaHostAlloc(void **p, size_t n, cudaHostAllocFlags)
{
    *p = new double[(n + 7) / 8];
    return cudaSuccess;
}
cudaError_t cudaGetLastError() { return cudaSuccess; }
cudaError_t cudaFree(void *p)
{
    delete[](double *) p;
    return cudaSuccess;
}
template <typename T>
cudaError_t cudaMalloc(T **p, size_t n)
{
    (*p) = (T *)new double[(n + 7) / 8];
    return cudaSuccess;
}
#endif

inline void checkCuda(cudaError_t result, const char *file, const int line)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "%s:%d: CUDA Runtime Error %d: %s\n", file, line, int(result), cudaGetErrorString(result));
        exit(-1);
    }
}
#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);
