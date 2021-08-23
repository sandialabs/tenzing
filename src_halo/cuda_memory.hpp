#pragma once

#include "sched/cuda_runtime.h"

#include <memory>

template <typename T>
struct Deleter {
    void operator()(T *p) {
        CUDA_RUNTIME(cudaFree(p));
    }
};

template <typename T>
std::shared_ptr<T> cuda_make_shared(const size_t n) {
    T *p;
    CUDA_RUNTIME(cudaMalloc(&p, n * sizeof(T)));
    return std::shared_ptr<T>(p, Deleter<T>());
}