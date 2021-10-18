#pragma once

#include "mcts_node.hpp"
#include "mcts_strategy.hpp"

namespace mcts {
/* score all children equally
*/
struct Unvisited {

    using MyNode = Node<Unvisited>;

    // track which child is associated with each parent for this traversal
    struct Context : public StrategyContext {};

    struct State : public StrategyState {
        std::vector<double> times;
    };

    // assign a value proportional to how many children the child has
    static double select(Context &, const MyNode &, const MyNode &child) {
        if (child.state_.times.empty()) return std::numeric_limits<double>::infinity();
        else return 0;
    }

    static void backprop(Context &, MyNode &node, const Benchmark::Result &br) {
        double elapsed = br.pct10;
        node.state_.times.push_back(elapsed);
    }
};
} // namespace mcts