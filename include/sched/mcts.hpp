#pragma once

#include "graph.hpp"

#include "mpi.h"

#include <vector>

namespace mcts {

struct SimResult {
    std::vector<std::shared_ptr<CpuNode>> path; // path that is simulated
    std::vector<double> times; // times from the simulation
};

struct Result {
    std::vector<SimResult> simResults;
};

/* use monte-carlo tree search to explore the space of stream assignments
*/
Result mcts(const Graph<CpuNode> &g, MPI_Comm comm);
} // namespace mcts