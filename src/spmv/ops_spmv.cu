/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "tenzing/mpi/ops_mpi.hpp"
#include "tenzing/spmv/ops_spmv.cuh"


void VectorAdd::run(cudaStream_t /*stream*/) {
  #warning VectorAdd::run(cudaStream_t) is a no-op
};