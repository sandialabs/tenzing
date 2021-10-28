#include "sched/ops_mpi.hpp"

#include "sched/macro_at.hpp"

#include <iostream>

void Irecv::run(Platform &/*plat*/)
{
    if (!args_.buf) THROW_RUNTIME("no buf");
    if (!args_.request) THROW_RUNTIME("no request");
    MPI_Irecv(args_.buf, args_.count, args_.datatype, args_.source, args_.tag, args_.comm, args_.request);
}

void Isend::run(Platform &/*plat*/)
{
    if (!args_.buf) THROW_RUNTIME("no buf");
    if (!args_.request) THROW_RUNTIME("no request");
    MPI_Isend(args_.buf, args_.count, args_.datatype, args_.dest, args_.tag, args_.comm, args_.request);
}

void Wait::run(Platform &/*plat*/)
{
    if (!args_.request) THROW_RUNTIME("Wait op has no request");
    MPI_Wait(args_.request, args_.status);
}

void OwningWaitall::run(Platform &/*plat*/)
{
    MPI_Waitall(reqs_.size(), reqs_.data(), MPI_STATUSES_IGNORE);
}


void MultiWait::run(Platform &/*plat*/)
{
    for (auto preq : reqs_) {
        MPI_Wait(preq, MPI_STATUS_IGNORE);
    }
}