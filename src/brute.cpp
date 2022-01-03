#include "sched/brute.hpp"

namespace brute {

std::vector<Sequence<BoundOp>> get_all_sequences(const Graph<OpBase> &g, Platform &plat) {
  std::vector<State> worklist;
  std::vector<Sequence<BoundOp>> ret;

  auto boundStart = std::dynamic_pointer_cast<BoundOp>(g.start());
  if (!boundStart) {
    THROW_RUNTIME("");
  }

  State initial;
  initial.graph = g;
  initial.sequence = {boundStart};
  worklist.push_back(initial);

  while (!worklist.empty()) {

    STDERR("get_all_sequences: worklist " << worklist.size() << " complete " << ret.size());

    // if (ret.size() >= 38) {
    //   break;
    // }

    // DFS
    State curr = worklist.back();
    worklist.pop_back();

    // get the frontier from the current state
    std::vector<std::shared_ptr<BoundOp>> frontier =
        mcts::get_graph_frontier(plat, curr.graph, curr.sequence, true);
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
        STDERR("get_all_sequences: add worklist entries for " << op->desc());

        // track if the child implies any platform binding
        STDERR("get_all_sequences: create graph by replacing unbound with " << op->desc());
        Graph<OpBase> cGraph = mcts::bind_unbound_vertex(curr.graph, op);

        auto syncs = mcts::get_syncs_before_op(cGraph, curr.sequence, op);
        if (!syncs.empty()) {
          STDERR("get_all_sequences: " << op->desc() << " required syncs.");
          for (const auto &cSync : syncs) {
            State next;
            next.graph = cGraph;
            next.sequence = curr.sequence;
            next.sequence.push_back(cSync);
            STDERR("add to worklist: " << get_desc_delim(next.sequence, ","));
            worklist.push_back(next);
          }
        } else {
          State next;
          next.graph = cGraph;
          next.sequence = curr.sequence;
          next.sequence.push_back(op);
          STDERR("add to worklist: " << get_desc_delim(next.sequence, ","));
          worklist.push_back(next);
        }
      }
    }
  }

  return ret;
}

void Result::dump_csv() const {

  const std::string delim("|");

  for (size_t i = 0; i < simResults.size(); ++i) {
    const auto &simres = simResults[i];
    std::cout << i;
    std::cout << delim << simres.benchResult.pct01;
    std::cout << delim << simres.benchResult.pct10;
    std::cout << delim << simres.benchResult.pct50;
    std::cout << delim << simres.benchResult.pct90;
    std::cout << delim << simres.benchResult.pct99;
    std::cout << delim << simres.benchResult.stddev;
    for (const auto &op : simres.seq) {
      std::cout << "|" << op->json();
    }
    std::cout << "\n";
  }
}

} // namespace brute