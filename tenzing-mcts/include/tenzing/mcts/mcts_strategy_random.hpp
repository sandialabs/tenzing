/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include "mcts_node.hpp"
#include "mcts_strategy.hpp"

#include <limits>
#include <map>

namespace tenzing::mcts {
/* score all children equally
 */
struct Random {

  using MyNode = Node<Random>;

  // track which child is associated with each parent for this traversal
  struct Context : public StrategyContext {
    std::map<const MyNode *, size_t> selected;
  };

  struct State : public StrategyState {
    std::vector<double> times;
  };

  // assign a value proportional to how many children the child has
  static double select(Context &ctx, const MyNode &child) {

    const MyNode &parent = *child.parent_;

    if (0 == ctx.selected.count(&parent)) {
      ctx.selected[&parent] = rand() % parent.children_.size();
    }
    size_t selected = ctx.selected.at(&parent);
    if (&child == &parent.children_[selected]) {
      return std::numeric_limits<double>::infinity();
    } else {
      return 0;
    }
  }

  static void backprop(Context &ctx, MyNode &node, const Benchmark::Result &br) {
    double elapsed = br.pct10;
    node.state_.times.push_back(elapsed);

    if (!node.parent_) {
      // once backprop to root, clear assignment before next traversal
      ctx.selected.clear();
    }
  }
};
} // namespace tenzing::mcts