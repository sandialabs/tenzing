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
    int flag;\
    MPI_Initialized(&flag);\
    if (flag) {\
        int rank;\
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);\
        std::cerr << "[" << rank << "] ";\
    } else {\
        std::cerr << "[x] ";\
    }\
    std::cerr << __FILE__ << ":" << __LINE__ << ": " << msg << "\n";\
}
