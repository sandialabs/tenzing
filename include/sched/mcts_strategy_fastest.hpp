#pragma once

#include <limits>
#include <algorithm>

#include "mcts_node.hpp"
#include "mcts_strategy.hpp"

namespace mcts {
/* score node higher if it's fastest run is faster
*/
struct PreferFastest {

    using MyNode = Node<PreferFastest>;

    struct Context : public StrategyContext {
        double minT;
        double maxT;
        Context() : minT(std::numeric_limits<double>::infinity()),
        maxT(-std::numeric_limits<double>::infinity()) {}
    };

    struct State : public StrategyState {
        std::vector<double> times;
    };

    // score child
    static double select(const Context &ctx, const MyNode &/*parent*/, const MyNode &child) {
        double v;
        if (child.state_.times.empty()) {
            v =  0;
        } else {
            double acc = child.state_.times[child.state_.times.size() * 0 / 100];
            v = (acc - ctx.minT) / (ctx.maxT - ctx.minT);
            v = 1-v;
            if (v < 0) v = 0;
            if (v > 1) v = 1;
        }
        return v;
    }

    static void backprop(Context &ctx, MyNode &node, const Schedule::BenchResult &br) {
        double elapsed = br.pct10;
        node.state_.times.push_back(elapsed);

        // order times smallest to largest
        std::sort(node.state_.times.begin(), node.state_.times.end());

        // keep track of a window of central values to compare speeds against
        if (!node.parent_) {
            size_t loi = node.state_.times.size() * 0 / 100;
            size_t hii = node.state_.times.size() * 100 / 100 - 1;
            ctx.minT = node.state_.times[loi];
            ctx.maxT = node.state_.times[hii];
        }
    }
};
} // namespace mcts