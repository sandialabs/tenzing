#include "sched/ops_mpi.hpp"

#include "sched/macro_at.hpp"

#include <iostream>

void Irecv::run()
{
    // std::cerr << "Irecvs...\n";
    // if (!args.buf) throw std::runtime_error(AT);
    // if (!args.request) throw std::runtime_error(AT);
    MPI_Irecv(args_.buf, args_.count, args_.datatype, args_.source, args_.tag, args_.comm, args_.request);
    // std::cerr << "Irecvs done\n";
}

void Isend::run()
{
    // std::cerr << "Isends...\n";
    // if (!args.buf) throw std::runtime_error(AT);
    // if (!args.request) throw std::runtime_error(AT);
    MPI_Isend(args_.buf, args_.count, args_.datatype, args_.dest, args_.tag, args_.comm, args_.request);
    // std::cerr << "Isends done\n";
}

void Wait::run()
{
    // std::cerr << "wait(Irecvs)...\n";
    // if (!args.request) throw std::runtime_error(AT);
    MPI_Wait(args_.request, args_.status);
    // std::cerr << "wait(Irecvs) done\n";
}