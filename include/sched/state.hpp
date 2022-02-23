#pragma once

#include "graph.hpp"
#include "platform.hpp"
#include "sequence.hpp"
#include "decision.hpp"

#include <vector>

/*! \brief a state in the sequential decision process
 */
class State {
  Graph<OpBase> graph_;
  Sequence<BoundOp> sequence_;

public:
  State(const Graph<OpBase> &graph, const Sequence<BoundOp> &sequence)
      : graph_(graph), sequence_(sequence) {}

  const Sequence<BoundOp> &sequence() const {return sequence_; }

  /*! \brief return the state resulting applying decision to this state
   */
  State apply(const Decision &d);

  /*! \brief return the unique states resulting from all possible decisions
   */
  std::vector<State> frontier(Platform &plat, bool quiet = true);
};

