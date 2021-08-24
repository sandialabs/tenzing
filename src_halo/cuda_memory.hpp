#pragma once

#include "sched/cuda_runtime.h"

#include <memory>

struct CudaFreeDeleter {
    void operator()(void *p) {
        CUDA_RUNTIME(cudaFree(p));
    }
};

/* return a shared_ptr to an allocation with space for n elements of T
   shared_ptr will manage lifetime of this allocation
*/
template <typename T>
std::shared_ptr<T> cuda_make_shared(const size_t n) {
    T *p;
    CUDA_RUNTIME(cudaMalloc(&p, n * sizeof(T)));
    return std::shared_ptr<T>(p, CudaFreeDeleter());
}