#pragma once

#include "mcts_node.hpp"
#include "mcts_strategy.hpp"

namespace mcts {

/* score child higher if it is anticorrelated with parent
*/
struct AntiCorrelation {

    using MyNode = Node<AntiCorrelation>;

    struct Context : public StrategyContext {}; // unused
    struct State : public StrategyState {
        std::vector<double> times;
    };
    const static int nBins = 10;

    static std::vector<uint64_t> histogram(
        const std::vector<double> &v,
        const double tMin, // low end of small bin
        const double tMax // high end of large bin
        ) {
            std::vector<uint64_t> hist(nBins, 0);

            for (double e : v) {
                size_t i = (e - tMin) / (tMax - tMin) * nBins;
                if (i >= nBins) i = nBins - 1;
                ++hist[i];
            }
            return hist;
    }

    // assign a value proportional to how much of the
    // space between the slowest and fastest run this child represents
    static double select(const Context &, const MyNode &parent, const MyNode &child) {
        double v;
        if (parent.state_.times.size() < 2 || child.state_.times.size() < 2) {
            v = 0;
        } else {

            double tMin = std::min(*parent.state_.times.begin(), *child.state_.times.begin());
            double tMax = std::max(parent.state_.times.back(), child.state_.times.back());

            // score children by inverse correlation with parent
            auto pHist = histogram(parent.state_.times, tMin, tMax);
            auto cHist = histogram(child.state_.times, tMin, tMax);
            v = corr(pHist, cHist); // [-1, 1]

            {
                std::stringstream ss;
                for (const auto &e : pHist) {
                    ss << e << " ";
                }
                STDERR(ss.str());
            }
            {
                std::stringstream ss;
                for (const auto &e : cHist) {
                    ss << e << " ";
                }
                STDERR(ss.str());
            }

            v += 1; // [0, 2]
            v = 2 - v; // prefer lower correlation [0,2]
            v /= 2; // [0,1]
        }
        if (v < 0) v = 0;
        if (v > 1) v = 1;
        return v;
    }

    static void backprop(Context &, MyNode &node, const Schedule::BenchResult &br) {
        double elapsed = br.pct10;
        node.state_.times.push_back(elapsed);

        // order times smallest to largest
        std::sort(node.state_.times.begin(), node.state_.times.end());
    }
};

} // namespace mcts