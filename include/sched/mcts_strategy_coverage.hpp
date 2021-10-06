#pragma once

#include "mcts.hpp"

namespace mcts {
/* score node higher if it's slow-fast range is wider
*/
struct Coverage {

    using MyNode = Node<Coverage>;

    struct Context : public StrategyContext {
        double minT;
        double maxT;
        Context() : minT(std::numeric_limits<double>::infinity()),
        maxT(-std::numeric_limits<double>::infinity()) {}
    };

    const static int loPct = 0;
    const static int hiPct = 100;

    // assign a value proportional to how much of the parent's slow-fast distance
    // the child covers
    static double select(const Context &ctx, const MyNode & parent, const MyNode &child) {
        double v;
        if (parent.times_.size() < 2 || child.times_.size() < 2) {
            // none or 1 measurement doesn't tell anything
            // about how fast this program is relative
            // to the overall
            v = 0;
            // prefer children that do not have enough runs yet
            // v = std::numeric_limits<double>::infinity();
        } else {
            double cMax = child.times_[child.times_.size() * hiPct / 100 - 1];
            double cMin = child.times_[child.times_.size() * loPct / 100];
            double pMax = parent.times_[parent.times_.size() * hiPct / 100 - 1];
            double pMin = parent.times_[parent.times_.size() * loPct / 100];
            v = (cMax - cMin) / (pMax - pMin);
            // v = 300.0 * stddev(child.times_) / avg(child.times_);
        }
        if (v < 0) v = 0;
        if (v > 1) v = 1;
        return v;
    }

    static void backprop(Context &ctx, MyNode &node, const Schedule::BenchResult &br) {

        double elapsed = br.pct10;
        node.times_.push_back(elapsed);

        // order times smallest to largest
        std::sort(node.times_.begin(), node.times_.end());

        // tell my parent to do the same
        if (node.parent_) {
            backprop(ctx, *node.parent_, br);
        }
        // keep track of a window of central values to compare speeds against
        else {
            size_t loi = node.times_.size() * loPct / 100;
            size_t hii = node.times_.size() * hiPct / 100 - 1;
            ctx.minT = node.times_[loi];
            ctx.maxT = node.times_[hii];
        }
    }
};
} // namespace mcts