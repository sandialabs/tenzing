/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

/*! \file brute-force search of the state space
 */

#pragma once

#include "tenzing/benchmarker.hpp"
#include "tenzing/cuda/ops_cuda.hpp"
#include "tenzing/graph.hpp"
#include "tenzing/operation.hpp"
#include "tenzing/sequence.hpp"
#include "tenzing/state.hpp"
#include "tenzing/trap.hpp"

#include <nlohmann/json.hpp>

#include <vector>

namespace tenzing {
namespace dfs {

struct Opts {
  int64_t maxSeqs; /// generate no more than this many sequences during DFS traversal. Some of these
                   /// sequences may be equivalent. Negative numbers mean unlimited.
  Benchmark::Opts benchOpts;

  Opts() : maxSeqs(-1) {}
};
void to_json(nlohmann::json &j, const Opts &opts);

struct SimResult {
  Sequence<BoundOp> seq;         // path that is simulated
  Benchmark::Result benchResult; // times from the simulation
};

struct Result {
  std::vector<SimResult> simResults;
  Opts opts_; /// options used to generate this result
  void dump_csv() const; // dump CSV to stdout

  Result() = delete;
  Result(const Opts &opts) : opts_(opts) {}
};

/* a stop signal to share between ranks */
struct Stop {
  enum class Reason { predicate };
  Reason reason_;
  bool value_;
  Stop() : value_(false) {}
  Stop(bool b, const Reason &reason) : reason_(reason), value_(b) {}
  explicit operator bool() const { return value_; }

  const char *c_str() const {
    switch (reason_) {
    case Reason::predicate:
      return "predicate";
    }
    THROW_RUNTIME("unexpected reason");
  }

  void bcast(int root, MPI_Comm comm) {
    MPI_Bcast(&value_, sizeof(value_), MPI_BYTE, root, comm);
    MPI_Bcast(&reason_, sizeof(reason_), MPI_BYTE, root, comm);
  }
};

std::vector<Sequence<BoundOp>> get_all_sequences(
    const Graph<OpBase> &g, Platform &plat,
    int64_t maxSeqs =
        -1 /// return once this many sequences have been generated. Negative is unlimited.
);

template <typename Benchmarker>
Result explore(const Graph<OpBase> &g, Platform &plat, Benchmarker &benchmarker,
               const Opts &opts = Opts()) {

  int rank = 0, size = 1;
  MPI_Comm_rank(plat.comm(), &rank);
  MPI_Comm_size(plat.comm(), &size);

  Result res(opts);

  std::vector<Sequence<BoundOp>> seqs;
  if (0 == rank) {
    // generate all sequences
    seqs = get_all_sequences(g, plat, opts.maxSeqs);

    // remove equivalent sequences
    STDERR("remove equivalent sequences");
    size_t removed = 0;
    for (auto si = seqs.begin(); si < seqs.end(); ++si) {
      for (auto sj = si + 1; sj < seqs.end(); ++sj) {
        Equivalence eqv = get_equivalence(*si, *sj);
        if (eqv) {

          STDERR("removed " << si - seqs.begin() + removed << " (= " << sj - seqs.begin() + removed
                            << "): " << get_desc_delim(*si, ", ") << " = "
                            << get_desc_delim(*sj, ", ") << " eq: " << eqv.str());

          ++removed;
          // si is before sj, so reset sj as well. sj is about to be incremented
          si = seqs.erase(si);
          sj = si;
        }
      }
    }
    STDERR("benchmark " << seqs.size() << " sequences");
  }

  // prevent a zillion cudaEventCreate calls
  CudaEventPool eventPool;

  // print results so far if interrupted
  std::function<void(int)> printResults = [&res](int /*sig*/) -> void { res.dump_csv(); };
  if (0 == rank) {
    register_handler(printResults);
  }

  // benchmark all sequences
  size_t i = 0;
  while (true) {

    Sequence<BoundOp> sut;
    Stop stop;
    if (0 == rank && i >= seqs.size()) {
      stop = Stop(true, Stop::Reason::predicate);
    }
    stop.bcast(0, plat.comm());

    if (stop) {
      STDERR("got stop: " << stop.c_str());
      break;
    }

    if (0 == rank) {
      sut = seqs[i++];
    }
    sut = mpi_bcast(sut, g, plat.comm());

    // provision resources for this program
    {
      eventPool.reset();
      ResourceMap rMap;
      {
        std::vector<HasEvent *> ops;

        for (const auto &op : sut) {
          if (HasEvent *he = dynamic_cast<HasEvent *>(op.get())) {
            ops.push_back(he);
          }
        }

        for (HasEvent *op : ops) {
          for (Event event : op->get_events()) {
            if (!rMap.contains(event)) {
              rMap.insert(event, eventPool.new_event());
            }
          }
        }
      }
      plat.resource_map() = rMap;
    }

    // STDERR("benchmark");

    SimResult sr;
    sr.seq = sut;
    sr.benchResult = benchmarker.benchmark(sut, plat, opts.benchOpts);
    res.simResults.push_back(sr);
  }
  unregister_handler();
  return res;
}
} // namespace dfs
} // namespace tenzing
