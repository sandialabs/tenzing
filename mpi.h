#pragma once

#include <mpi.h>

struct IrecvArgs
{
    void *buf;
    int count;
    MPI_Datatype datatype;
    int source;
    int tag;
    MPI_Comm comm;
    MPI_Request *request;
    bool operator==(const IrecvArgs &rhs) const {
        return buf == rhs.buf && count == rhs.count && datatype == rhs.datatype && source == rhs.source && tag == rhs.tag && comm == rhs.comm && request == rhs.request;
    }
};

struct IsendArgs
{
    const void *buf;
    int count;
    MPI_Datatype datatype;
    int dest;
    int tag;
    MPI_Comm comm;
    MPI_Request *request;
    bool operator==(const IsendArgs &rhs) const {
        return buf == rhs.buf && count == rhs.count && datatype == rhs.datatype && dest == rhs.dest && tag == rhs.tag && comm == rhs.comm && request == rhs.request;
    }
};
struct WaitArgs
{
    MPI_Request *request;
    MPI_Status *status;
        bool operator==(const WaitArgs &rhs) const {
        return request == rhs.request && status == rhs.status;
    }
};
