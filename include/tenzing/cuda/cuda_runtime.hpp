/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

/* Define some things that might help a C++ compiler
   compile (non-functional) CUDA code
*/

#pragma once

#include <cstdio>
#include <cstdlib>
#include <ostream>

#include <cuda_runtime.h>
#include <cusparse.h>


inline void checkCuda(cudaError_t result, const char *file, const int line)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "%s:%d: CUDA Runtime Error %d: %s\n", file, line, int(result), cudaGetErrorString(result));
        exit(-1);
    }
}
#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);

inline void checkCusparse(cusparseStatus_t result, const char *file, const int line)
{
    if (result != CUSPARSE_STATUS_SUCCESS)
    {
        fprintf(stderr, "%s:%d: cuSPARSE Error %d: %s\n", file, line, int(result), cusparseGetErrorString(result));
        exit(-1);
    }
}
#define CUSPARSE(stmt) checkCusparse(stmt, __FILE__, __LINE__);

inline std::ostream& operator<<(std::ostream& os, cudaStream_t s)
{
    os << uintptr_t(s);
    return os;
}