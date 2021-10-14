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
        bool fullyVisited;         // true if every child node has been visited
        std::vector<double> times; // the times of all runs from this node 
        State() : fullyVisited(false) {}
    };

    const static int loPct = 0;
    const static int hiPct = 100;

    // assign a value proportional to how much of the parent's slow-fast distance
    // the child covers
    static double select(const Context &, const MyNode & parent, const MyNode &child) {
        double v;
        if (parent.state_.times.size() < 2 || child.state_.times.size() < 2) {
            // none or 1 measurement doesn't tell anything
            // about how fast this program is relative
            // to the overall
            v = 0;
            // prefer children that do not have enough runs yet
            // v = std::numeric_limits<double>::infinity();
        } else {
            double cMax = child.state_.times[child.state_.times.size() * hiPct / 100 - 1];
            double cMin = child.state_.times[child.state_.times.size() * loPct / 100];
            double pMax = parent.state_.times[parent.state_.times.size() * hiPct / 100 - 1];
            double pMin = parent.state_.times[parent.state_.times.size() * loPct / 100];
            v = (cMax - cMin) / (pMax - pMin);
            // v = 300.0 * stddev(child.state_.times) / avg(child.state_.times);
        }
        if (v < 0) v = 0;
        if (v > 1) v = 1;
        return v;
    }

    static void backprop(Context &ctx, MyNode &node, const Schedule::BenchResult &br) {

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