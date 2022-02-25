/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include <limits>
#include <algorithm>

#include "mcts_node.hpp"
#include "mcts_strategy.hpp"

namespace mcts {
/* map root node times into range [1, 0]
   figure out where each of node times falls in there, and do weighted average
*/
struct AvgTime {

    using MyNode = Node<AvgTime>;

    struct Context : public StrategyContext {};

    struct State : public StrategyState {
        std::vector<double> times;
        double tMin;
        double tMax;
        State() : tMin(std::numeric_limits<double>::infinity()), tMax(-std::numeric_limits<double>::infinity()) {}
    };

    static float select(const Context &/*ctx*/, const MyNode &/*parent*/, const MyNode &child) {

        const MyNode &root = child.root();

        if (child.n_ < 1 || root.n_ < 2) {
            return 0;
        } else {
            double acc  = 0;
            for (double t : child.state_.times) {
                double v  = (t - root.state_.tMin) / (root.state_.tMax - root.state_.tMin);
                v = 1-v;
                if (v < 0) v = 0;
                if (v > 1) v = 1;
                acc += v;
            }
            return acc / child.state_.times.size();
        }
    }

    static void backprop(Context &/*ctx*/, MyNode &node, const Benchmark::Result &br) {
        node.state_.tMin = std::min(br.pct10, node.state_.tMin);
        node.state_.tMax = std::max(br.pct10, node.state_.tMax);
        node.state_.times.push_back(br.pct10);
    }
};


inline std::ostream &operator<<(std::ostream &os, const AvgTime::State &s) {
    os << "[" << s.tMin << ", " << s.tMax << "]";
    return os;
}


} // namespace mcts

