#pragma once

#include <limits>

#include "mcts_node.hpp"
#include "mcts_strategy.hpp"

namespace mcts {
/* score node higher if it's slow-fast range is wider
*/
struct Coverage {

    using MyNode = Node<Coverage>;

    // Context globally available during MCTS
    struct Context : public StrategyContext {
        double minT;
        double maxT;
        Context() : minT(std::numeric_limits<double>::infinity()),
        maxT(-std::numeric_limits<double>::infinity()) {}
    };

    // State of the node
    struct State : public StrategyState {
        std::vector<double> times; // the times of all runs from this node 
    };

    const static int loPct = 0;
    const static int hiPct = 100;

    // assign a value proportional to how much of the parent's slow-fast distance
    // the child covers
    static double select(const Context &, const MyNode &parent, const MyNode &child) {

        if (parent.state_.times.size() < 2) {
            return 1; // if the parent doesn't have enough runs, assume the child just covers it
        } else if (child.state_.times.size() < 1) {
            // if the child has no runs, assume the child covers the parent

            // FIXME, this should be the parent's runs at the time
            return 1;
        } else if (child.state_.times.size() < 2) {
            double pMax = parent.state_.times[parent.state_.times.size() * hiPct / 100 - 1];
            double pMin = parent.state_.times[parent.state_.times.size() * loPct / 100];

            // parent min and max may represent the same rollout and get the same time
            if (pMin == pMax) {
                return 1;
            }

            double v = std::max(child.state_.times[0] - pMin, pMax - child.state_.times[0]) / (pMax - pMin);
            if (v < 0) v = 0;
            if (v > 1) v = 1;
            return v;
        } else {
            double cMax = child.state_.times[child.state_.times.size() * hiPct / 100 - 1];
            double cMin = child.state_.times[child.state_.times.size() * loPct / 100];
            double pMax = parent.state_.times[parent.state_.times.size() * hiPct / 100 - 1];
            double pMin = parent.state_.times[parent.state_.times.size() * loPct / 100];

            // parent min and max may represent the same rollout and get the same time
            if (pMin == pMax) {
                return 1;
            }

            double v = (cMax - cMin) / (pMax - pMin);
            if (v < 0) v = 0;
            if (v > 1) v = 1;
            return v;
        }

    }

    static void backprop(Context &ctx, MyNode &node, const Benchmark::Result &br) {

        double elapsed = br.pct10;
        node.state_.times.push_back(elapsed);

        // order times smallest to largest
        std::sort(node.state_.times.begin(), node.state_.times.end());

        // keep track of a window of central values to compare speeds against
        if (!node.parent_) {
            size_t loi = node.state_.times.size() * loPct / 100;
            size_t hii = node.state_.times.size() * hiPct / 100 - 1;
            ctx.minT = node.state_.times[loi];
            ctx.maxT = node.state_.times[hii];
        }
    }
};
} // namespace mcts