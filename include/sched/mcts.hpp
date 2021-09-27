#pragma once

#include "graph.hpp"
#include "schedule.hpp"

#include "mpi.h"

#include <vector>

namespace mcts {

struct SimResult {
    std::vector<std::shared_ptr<CpuNode>> path; // path that is simulated
    Schedule::BenchResult benchResult; // times from the simulation
};

struct Result {
    std::vector<SimResult> simResults;
};

struct Opts {
    size_t dumpTreeEvery; // how often to dump the tree
    std::string dumpTreePrefix; // prefix to use for the tree
    BenchOpts benchOpts; // options for the benchmark runs

    Opts() : dumpTreeEvery(100) {}
};

/* use monte-carlo tree search to explore the space of stream assignments
*/
Result mcts(const Graph<CpuNode> &g, MPI_Comm comm, const Opts &opts = Opts());

} // namespace mcts