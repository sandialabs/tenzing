#include <doctest/doctest.hpp>

#include <iostream>

#include "tenzing/graph.hpp"
#include "tenzing/spmv/ops_spmv.cuh"
#include "tenzing/state.hpp"

TEST_CASE("expand spmv") {

  MPI_Init(nullptr, nullptr);
  STDERR("finished MPI_Init()");

  typedef int Ordinal;
  typedef float Scalar;
  int m = 150; // 150 x 150 matrix
  int size = 4; // 4 ranks
  SpMV<Ordinal, Scalar>::Opts spmvOpts {
    .m = m,
    .bw = m/size,
    .nnz = m * 10
  };

  auto spmv = std::make_shared<SpMV<Ordinal, Scalar>>(spmvOpts, MPI_COMM_WORLD);

  Graph<OpBase> orig;
  CHECK(orig.vertex_size() == 2);
  orig.start_then(spmv);
  orig.then_finish(spmv);

  SDP::State initial(orig);
  SDP::State final = initial.apply(ExpandOp(spmv));

  CHECK(final.graph().vertex_size() == spmv->graph().vertex_size());

  spmv->graph().dump_graphviz("test_spmv.dot");
  final.graph().dump_graphviz("test_final.dot");

  MPI_Finalize();
}