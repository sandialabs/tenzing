/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */


#include "tenzing/spmv/ops_spmv.cuh"

void Scatter::run(cudaStream_t stream) {
  #ifdef SANITY_CHECKS
      if (args_.dst.size() != args_.idx.size()) {
        THROW_RUNTIME("scatter dst size was different than index size");
      }
      if (!args_.dst.data_) {
          THROW_RUNTIME("bad dst");
      }
      if (!args_.src.data_) {
          THROW_RUNTIME("bad src");
      }
      if (!args_.idx.data_) {
          THROW_RUNTIME("bad idx");
      }
  #endif
      scatter<<<128, 100, 0, stream>>>(args_.dst, args_.src, args_.idx);
      CUDA_RUNTIME(cudaGetLastError());
    }

void VectorAdd::run(cudaStream_t /*stream*/) {
  #warning VectorAdd::run(cudaStream_t) is a no-op
};
