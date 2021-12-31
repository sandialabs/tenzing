#pragma once

#include <limits>
#include <algorithm>

#include "mcts_node.hpp"
#include "mcts_strategy.hpp"

namespace mcts {
/* score a node proportional to how close its minimum time is to the overall
*/
struct MinTime {

    using MyNode = Node<MinTime>;

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
    static float select(const Context &/*ctx*/, const MyNode &/*parent*/, const MyNode &child) {

        const MyNode &root = child.root();

        if (child.n_ < 1 || root.n_ < 2) {
            return 0;
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


inline std::ostream &operator<<(std::ostream &os, const MinTime::State &s) {
    os << "[" << s.tMin << ", " << s.tMax << "]";
    return os;
}


} // namespace mcts

