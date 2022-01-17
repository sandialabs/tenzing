/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include <sstream>
#include <iostream>

#include <mpi.h>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__) 

#define THROW_RUNTIME(msg) \
{\
    std::stringstream _ss;\
    _ss << __FILE__ << ":" << __LINE__ << ": " << msg << "\n";\
    throw std::runtime_error(_ss.str());\
}


#define STDERR(msg) \
{\
    int xxStdErrFlag;\
    MPI_Initialized(&xxStdErrFlag);\
    if (xxStdErrFlag) {\
        int xxStdErrRank;\
        MPI_Comm_rank(MPI_COMM_WORLD, &xxStdErrRank);\
        std::cerr << "[" << xxStdErrRank << "] ";\
    } else {\
        std::cerr << "[x] ";\
    }\
    std::cerr << __FILE__ << ":" << __LINE__ << ": " << msg << "\n";\
}

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 1)