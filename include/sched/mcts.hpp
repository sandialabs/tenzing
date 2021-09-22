#pragma once

#include "graph.hpp"

namespace mcts {

/* use monte-carlo tree search to explore the space of stream assignments
*/
void mcts(const Graph<CpuNode> &g);
} // namespace mcts