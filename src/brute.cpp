#include "sched/brute.hpp"

namespace brute {

std::vector<Sequence<BoundOp>> get_all_sequences(const Graph<OpBase> &g, Platform &plat) {
  std::vector<State> worklist;
  std::vector<Sequence<BoundOp>> ret;

  State initial;
  initial.graph = g;
  initial.sequence = {};
  worklist.push_back(initial);

  while (!worklist.empty()) {
    State curr = worklist.back();
    worklist.pop_back();

    // get the frontier from the current state
    std::vector<std::shared_ptr<BoundOp>> frontier =
        mcts::get_graph_frontier(plat, curr.graph, curr.sequence);
    {
      std::string s;
      for (const auto &op : frontier) {
        s += op->desc();
        s += ", ";
      }
      STDERR("get_all_sequences: graph frontier is: " << s);
    }

    if (frontier.empty()) {
      ret.push_back(curr.sequence);
    } else {
      /* create child nodes in frontier
             if the child does not require synchronization, use it directly
             if it does, create one child for each possible synchronization
          */
      STDERR("create child nodes...");
      for (const std::shared_ptr<BoundOp> &op : frontier) {
        STDERR("get_all_sequences: create child node(s) for " << op->desc());

        // track if the child implies any platform binding
        STDERR("get_all_sequences: create graph by replacing unbound with " << op->desc());
        Graph<OpBase> cGraph = mcts::bind_unbound_vertex(curr.graph, op);

        auto syncs = mcts::get_syncs_before_op(plat, cGraph, curr.sequence, op);
        if (!syncs.empty()) {
          STDERR("get_all_sequences: " << op->desc() << " required syncs.");
          for (const auto &cSync : syncs) {
            State next;
            next.graph = cGraph;
            next.sequence = curr.sequence;
            next.sequence.push_back(cSync);
            worklist.push_back(next);
          }
        } else {
          State next;
          next.graph = cGraph;
          next.sequence = curr.sequence;
          next.sequence.push_back(op);
          worklist.push_back(next);
        }
      }
    }
  }

  return ret;
}

} // namespace brute