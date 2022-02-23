/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "sched/mpi/ops_mpi.hpp"
#include "sched/spmv/ops_spmv.cuh"


void VectorAdd::run(cudaStream_t /*stream*/) {
  #warning VectorAdd::run(cudaStream_t) is a no-op
};