#pragma once

#include "mcts.hpp"

namespace mcts {

/* score child higher if it is correlated with root. normalize with siblings
*/
struct NormRootCorr {

    using MyNode = Node<NormRootCorr>;

    struct Context : public StrategyContext {
        MyNode *root;
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
        if (parent.times_.size() < 2 || child.times_.size() < 2) {
            return 0;
        } else {

#if 0
            double tMin = *parent.times_.begin();
            double tMax = parent.times_.back();
            auto pHist = histogram(parent.times_, tMin, tMax);
#else
            double tMin = *ctx.root->times_.begin();
            double tMax = ctx.root->times_.back();
            auto pHist = histogram(ctx.root->times_, tMin, tMax);
#endif
            std::vector<double> anticorrs;

            // score children by inverse correlation with parent
            for (const MyNode &sib : parent.children_) {
                auto cHist = histogram(sib.times_, tMin, tMax);
                double c = corr(pHist, cHist); // [-1,1]
                c += 1; // [0,2]
                anticorrs.push_back(c);
            }

            // find max correlation to normalize
            double maxCorr = -1;
            for (double c : anticorrs) {
                maxCorr = std::max(c, maxCorr);
            }
            auto cHist = histogram(child.times_, tMin, tMax);

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

    static void backprop(Context &ctx, MyNode &node, const Schedule::BenchResult &br) {
        double elapsed = br.pct10;
        node.times_.push_back(elapsed);

        // order times smallest to largest
        std::sort(node.times_.begin(), node.times_.end());

        // tell my parent to do the same
        if (node.parent_) {
            backprop(ctx, *node.parent_, br);
        } else {
            ctx.root = &node;
        }
    }
};
} // namespace mcts