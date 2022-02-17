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
/* score a node proportional to how close its minimum time is to the root's minimum time
*/
struct FastMin {

    using MyNode = Node<FastMin>;

    struct Context : public StrategyContext {};

    struct State : public StrategyState {
        double tMin;
        double tMax;
        State() : tMin(std::numeric_limits<double>::infinity()), tMax(-std::numeric_limits<double>::infinity()) {}
        
        std::string graphviz_label_line() const {
            std::stringstream ss;
            ss << std::scientific;
            ss.precision(2); // 2 digits after decimal
            ss << tMin << " - " << tMax;
            return ss.str();
        }
    };

    // score child
    static float select(const Context &ctx, const MyNode &child) {

        const MyNode &root = child.root();

        if (&child == &root) {
            return 1;
        } else if (root.n_ < 2 || root.state_.tMax == root.state_.tMin) { // root doesn't have enough info to score
            return 1;
        } else if (child.n_ < 1) { // child doesn't have enough info to score, use parent score
            return select(ctx, *(child.parent_));
        } else {
            float v = (child.state_.tMin - root.state_.tMin) / (root.state_.tMax - root.state_.tMin);
            v = 1-v;
            if (v < 0) v = 0;
            if (v > 1) v = 1;
            return v;
        }
    }

    static void backprop(Context &/*ctx*/, MyNode &node, const Benchmark::Result &br) {
        node.state_.tMin = std::min(br.pct10, node.state_.tMin);
        node.state_.tMax = std::max(br.pct10, node.state_.tMax);
    }
};


inline std::ostream &operator<<(std::ostream &os, const FastMin::State &s) {
    os << "[" << s.tMin << ", " << s.tMax << "]";
    return os;
}


} // namespace mcts

