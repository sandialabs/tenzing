#pragma once

#include "mcts.hpp"

#include <map>

namespace mcts {
/* score all children equally
*/
struct Random {

    using MyNode = Node<Random>;

    // track which child is associated with each parent for this traversal
    struct Context : public StrategyContext {
        std::map<const MyNode* , size_t> selected;
    };

    // assign a value proportional to how many children the child has
    static double select(Context &ctx, const MyNode &parent, const MyNode &child) {

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

    static void backprop(Context &ctx, MyNode &node, const Schedule::BenchResult &br) {
        double elapsed = br.pct10;
        node.times_.push_back(elapsed);

        // tell my parent to do the same
        if (node.parent_) {
            backprop(ctx, *node.parent_, br);
        } else {
            // once backprop to root, clear assignment before next traversal
            ctx.selected.clear();
        }
    }
};
} // namespace mcts