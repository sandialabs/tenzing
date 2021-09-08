#pragma once

#include "graph.hpp"

namespace mcts {

/* use monte-carlo tree search to explore the space of stream assignments
*/
void mcts(Graph<CpuNode> &g);
} // namespace mcts