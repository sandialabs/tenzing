#pragma once

#ifndef REAL_CUDA
typedef int cudaStream_t;
typedef int cudaError_t;
cudaError_t cudaStreamCreate(cudaStream_t *stream) {
    static int num = 0;
    *stream=(++num); return 0;
}
#else 
#include <cuda_runtime.h>
#endif