#pragma once

#include "mcts_node.hpp"
#include "mcts_strategy.hpp"

namespace mcts {

/* score child higher if it is correlated with root. normalize with siblings
*/
struct NormRootCorr {

    using MyNode = Node<NormRootCorr>;

    struct Context : public StrategyContext {
        MyNode *root;
    };

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
    static double select(const Context &ctx, const MyNode &parent, const MyNode &child) {
        if (parent.state_.times.size() < 2 || child.state_.times.size() < 2) {
            return 0;
        } else {

#if 0
            double tMin = *parent.state_.times.begin();
            double tMax = parent.state_.times.back();
            auto pHist = histogram(parent.state_.times, tMin, tMax);
#else
            double tMin = *ctx.root->state_.times.begin();
            double tMax = ctx.root->state_.times.back();
            auto pHist = histogram(ctx.root->state_.times, tMin, tMax);
#endif
            std::vector<double> anticorrs;

            // score children by inverse correlation with parent
            for (const MyNode &sib : parent.children_) {
                auto cHist = histogram(sib.state_.times, tMin, tMax);
                double c = corr(pHist, cHist); // [-1,1]
                c += 1; // [0,2]
                anticorrs.push_back(c);
            }

            // find max correlation to normalize
            double maxCorr = -1;
            for (double c : anticorrs) {
                maxCorr = std::max(c, maxCorr);
            }
            auto cHist = histogram(child.state_.times, tMin, tMax);

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

            double c = corr(pHist, cHist); // [-1,1]
            c += 1; // [0,2]
            STDERR(c << " / " << maxCorr);
            return c / maxCorr;
        }
    }

    static void backprop(Context &ctx, MyNode &node, const Benchmark::Result &br) {
        double elapsed = br.pct10;
        node.state_.times.push_back(elapsed);

        // order times smallest to largest
        std::sort(node.state_.times.begin(), node.state_.times.end());

        // tell my parent to do the same
        if (!node.parent_) {
            ctx.root = &node;
        }
    }
};
} // namespace mcts