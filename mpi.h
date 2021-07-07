#pragma once

#ifdef REAL_MPI
#include <mpi.h>
#else
struct MPI_Datatype
{
};
struct MPI_Comm
{
};
struct MPI_Request
{
};
struct MPI_Status
{
};
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm, MPI_Request *request) { return 0; }
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
              int tag, MPI_Comm comm, MPI_Request *request) { return 0; }
int MPI_Wait(MPI_Request *request, MPI_Status *status) { return 0; }
int MPI_Init(int *argc, char ***argv) { return 0; }
int MPI_Finalize() { return 0; }
double MPI_Wtime() { return 0; }
MPI_Comm MPI_COMM_WORLD;
#endif

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
