#pragma once

#include "mcts.hpp"

namespace mcts {
/* score all children equally
*/
struct Unvisited {

    using MyNode = Node<Unvisited>;

    // track which child is associated with each parent for this traversal
    struct Context : public StrategyContext {};

    // assign a value proportional to how many children the child has
    static double select(Context &, const MyNode &, const MyNode &child) {
        if (child.times_.empty()) return std::numeric_limits<double>::infinity();
        else return 0;
    }

    static void backprop(Context &ctx, MyNode &node, const Schedule::BenchResult &br) {
        double elapsed = br.pct10;
        node.times_.push_back(elapsed);

        // tell my parent to do the same
        if (node.parent_) {
            backprop(ctx, *node.parent_, br);
        }
    }
};
} // namespace mcts