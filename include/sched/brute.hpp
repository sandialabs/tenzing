#pragma once

/*! \file brute-force search of the state space
 */

#include "benchmarker.hpp"
#include "graph.hpp"
#include "mcts_node.hpp"
#include "operation.hpp"
#include "ops_cuda.hpp"
#include "sequence.hpp"

#include <vector>

namespace brute {

struct Opts {
    Benchmark::Opts benchOpts;
};

struct State {
  Graph<OpBase> graph;
  Sequence<BoundOp> sequence;
};

struct SimResult {
  Sequence<BoundOp> seq;         // path that is simulated
  Benchmark::Result benchResult; // times from the simulation
};

struct Result {
  std::vector<SimResult> simResults;
  void dump_csv() const; // dump CSV to stdout
};

std::vector<Sequence<BoundOp>> get_all_sequences(const Graph<OpBase> &g, Platform &plat);

template <typename Benchmarker>
Result brute(const Graph<OpBase> &g, Platform &plat, Benchmarker &benchmarker,
             const Opts &opts = Opts()) {

  Result res;

  // generate all sequences
  std::vector<Sequence<BoundOp>> seqs = get_all_sequences(g, plat);

  // remove equivalent sequences

  // benchmark all sequences
  for (const Sequence<BoundOp> &seq : seqs) {
    Sequence<BoundOp> sut = mpi_bcast(seq, g, plat.comm());

    SimResult sr;
    sr.seq = sut;
    sr.benchResult = benchmarker.benchmark(sut, plat, opts.benchOpts);
    res.simResults.push_back(sr);
  }
  return res;
}
} // namespace brute
