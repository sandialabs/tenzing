/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "sched/mpi/ops_mpi.hpp"
#include "sched/macro_at.hpp"

#include <iostream>

void Irecv::run(Platform & /*plat*/) {
  if (!args_.buf)
    THROW_RUNTIME("no buf");
  if (!args_.request)
    THROW_RUNTIME("no request");
  MPI_Irecv(args_.buf, args_.count, args_.datatype, args_.source, args_.tag, args_.comm,
            args_.request);
}

void Isend::run(Platform & /*plat*/) {
  if (!args_.buf)
    THROW_RUNTIME("no buf");
  if (!args_.request)
    THROW_RUNTIME("no request");
  MPI_Isend(args_.buf, args_.count, args_.datatype, args_.dest, args_.tag, args_.comm,
            args_.request);
}

void Ialltoallv::run(Platform & /*plat*/) {
  if (!args_.request)
    THROW_RUNTIME("Ialltoallv op has no request");
  MPI_Ialltoallv(args_.sendbuf, args_.sendcounts, args_.sdispls, args_.sendtype, args_.recvbuf,
                 args_.recvcounts, args_.rdispls, args_.recvtype, args_.comm, args_.request);
}

void Wait::run(Platform & /*plat*/) {
  if (!args_.request)
    THROW_RUNTIME("Wait op has no request");
  MPI_Wait(args_.request, args_.status);
}

void OwningWaitall::run(Platform & /*plat*/) {
  MPI_Waitall(reqs_.size(), reqs_.data(), MPI_STATUSES_IGNORE);
}

void MultiWait::run(Platform & /*plat*/) {
  for (auto preq : reqs_) {
    MPI_Wait(preq, MPI_STATUS_IGNORE);
  }
}