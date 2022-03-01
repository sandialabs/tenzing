#include <doctest/doctest.hpp>

#include <iostream>

#include "tenzing/graph.hpp"
#include "tenzing/spmv/ops_spmv.cuh"
#include "tenzing/state.hpp"

typedef int Ordinal;
typedef float Scalar;

template <Where w>
using csr_type = CsrMat<w, Ordinal, Scalar>;

TEST_CASE("expand spmv") {

  MPI_Init(nullptr, nullptr);
  STDERR("finished MPI_Init()");


  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);


  int m = 150; // 150 x 150 matrix
  int size = 4; // 4 ranks
  int nnz = m * 10;
  int bw = m / size;


  csr_type<Where::host> A;

  // generate and distribute A
  if (0 == rank) {
    std::cerr << "generate matrix\n";
    A = random_band_matrix<Ordinal, Scalar>(m, bw, nnz);
  }

  RowPartSpmv<Ordinal, Scalar> rps(A, 0, MPI_COMM_WORLD);

  auto spmv = std::make_shared<SpMV<Ordinal, Scalar>>(rps, MPI_COMM_WORLD);

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