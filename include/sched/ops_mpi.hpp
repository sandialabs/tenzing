/* mpi-specific operations
*/

#pragma once

#include "operation.hpp"

#include <mpi.h>

class Irecv : public CpuNode
{
public:
    struct Args
    {
        void *buf;
        int count;
        MPI_Datatype datatype;
        int source;
        int tag;
        MPI_Comm comm;
        MPI_Request *request;
        bool operator==(const Args &rhs) const {
            return buf == rhs.buf && count == rhs.count && datatype == rhs.datatype && source == rhs.source && tag == rhs.tag && comm == rhs.comm && request == rhs.request;
        }
    };
    Args args_;
    Irecv(Args args) : args_(args) {}
    std::string name() const override { return "Irecv"; }
    virtual void run() override;

    virtual int tag() const override { return 5; }

    CLONE_DEF(Irecv);
    EQ_DEF(Irecv);
    LT_DEF(Irecv);
    bool operator==(const Irecv &rhs) const {
        return args_ == rhs.args_;
    }
    bool operator<(const Irecv &rhs) const {
        return name() < rhs.name();
    }
};



class Isend : public CpuNode
{
public:
    struct Args
    {
        const void *buf;
        int count;
        MPI_Datatype datatype;
        int dest;
        int tag;
        MPI_Comm comm;
        MPI_Request *request;
        bool operator==(const Args &rhs) const {
            return buf == rhs.buf && count == rhs.count && datatype == rhs.datatype && dest == rhs.dest && tag == rhs.tag && comm == rhs.comm && request == rhs.request;
        }
    };
    Args args_;
    Isend(Args args) : args_(args) {}
    std::string name() const override { return "Isend"; }

    virtual void run() override;

    virtual int tag() const override { return 6; }

    CLONE_DEF(Isend);
    EQ_DEF(Isend);
    LT_DEF(Isend);
    bool operator==(const Isend &rhs) const {
        return args_ == rhs.args_;
    }
    bool operator<(const Isend &rhs) const {
        return name() < rhs.name();
    }
};

class Wait : public CpuNode
{
public:
    struct Args {
        MPI_Request *request;
        MPI_Status *status;
            bool operator==(const Args &rhs) const {
            return request == rhs.request && status == rhs.status;
        }
    };
    Args args_;
    Wait(Args args) : args_(args) {}
    std::string name() const override { return "Wait"; }

    virtual void run() override;
    virtual int tag() const override { return 7; }

    CLONE_DEF(Wait);
    EQ_DEF(Wait);
    LT_DEF(Wait);
    bool operator==(const Wait &rhs) const {
        return args_ == rhs.args_;
    }
    bool operator<(const Wait &rhs) const {
        return name() < rhs.name();
    }
};