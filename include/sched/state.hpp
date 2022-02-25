#pragma once

#include "decision.hpp"
#include "event_synchronizer.hpp"
#include "graph.hpp"
#include "platform.hpp"
#include "sequence.hpp"

#include <vector>

namespace SDP {

/*! \brief a state in the sequential decision process
 */
class State {
  Graph<OpBase> graph_;
  Sequence<BoundOp> sequence_;
  typedef EventSynchronizer Synchronizer;

public:
  State(const Graph<OpBase> &graph, const Sequence<BoundOp> &sequence)
      : graph_(graph), sequence_(sequence) {}

  const Sequence<BoundOp> &sequence() const { return sequence_; }
  const Graph<OpBase> &graph() const { return graph_; }

  /*! \brief return any required synchronization operations needed between this state and `op`
   */
  std::vector<std::shared_ptr<BoundOp>>
  get_syncs_before_op(const std::shared_ptr<BoundOp> &op) const;

  /*! \brief Get all possible decisions available from this state
   */
  std::vector<std::shared_ptr<Decision>> get_decisions(Platform &plat) const;

  /*! \brief return the state resulting applying decision to this state
   */
  State apply(const Decision &d) const;

  /*! \brief return the unique states resulting from all possible decisions
   */
  std::vector<State> frontier(Platform &plat, bool quiet = true);
};

} // namespace SDP