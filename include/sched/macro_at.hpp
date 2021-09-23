#pragma once

#include <sstream>
#include <iostream>

#include <mpi.h>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__) 

#define THROW_RUNTIME(msg) \
{\
    std::stringstream ss;\
    ss << __FILE__ << ":" << __LINE__ << ": " << msg << "\n";\
    throw std::runtime_error(ss.str());\
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
