#pragma once

#include "mcts_node.hpp"
#include "mcts_strategy.hpp"

namespace mcts {
/* select child that is brings up smallest parent histogram bin
*/
struct BalanceHistogram {

    using MyNode = Node<BalanceHistogram>;

    struct Context : public StrategyContext {
        MyNode *root;
        
        Context() : root(nullptr) {}
    }; // unused

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

    // value children who have the most runs that can balance parent histogram bin sizes
    // choose the smallest (non-zero) parent bin
    // figure out which child has the largest proportion of its runs fall into that bin
    // score the child relative to that largest proportion number
    static double select(const Context &ctx, const MyNode &parent, const MyNode &child) {
        if (parent.state_.times.size() < 1 || child.state_.times.size() < 1) {
            return 0;
        } else {
#if 0
            double tMin = std::min(*parent.state_.times.begin(), *child.state_.times.begin());
            double tMax = std::max(parent.state_.times.back(), child.state_.times.back());

            tMin = *ctx.root->times_.begin();
            tMax = ctx.root->times_.back();
            auto rHist = histogram(ctx.root->times_, tMin, tMax);
            auto pHist = histogram(parent.state_.times, tMin, tMax);
            auto cHist = histogram(child.state_.times, tMin, tMax);




#if 1
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
#endif

            // smallest non-zero histogram bin
            int64_t smallest = -1;
            {
                int64_t cnt = std::numeric_limits<int64_t>::max();
                for (size_t i = 0; i < pHist.size(); ++i) {
                    if (rHist[i] > 0 && pHist[i] < cnt) {
                        smallest = i;
                        cnt = rHist[i];
                    }   
                }
            }
            // largest non-zero histogram bin
            int64_t largest = -1;
            {
                int64_t cnt = -1;
                for (size_t i = 0; i < pHist.size(); ++i) {
                    if (pHist[i] > 0 && pHist[i] > cnt) {
                        largest = i;
                        cnt = pHist[i];
                    }   
                }
            }


            // score child by count in that bin / total count
            if (-1 == smallest) {
                // no bin is non-zero
                return 0;
            } else {

                // across all children, determine which has the largest proportion of runs in that bin
                double maxProp = -1;
                for (const auto &sib : parent.children_) {
                    auto ccHist = histogram(sib.times_, tMin, tMax);
                    double prop = double(ccHist[smallest]) / sib.times_.size();
                    if (prop > maxProp) {
                        maxProp = prop;
                    }
                }

                // weight count compared to max sibling count
                if (0 == maxProp) {
                    return 0;
                } else {
                    return double(cHist[smallest]) / double(child.state_.times.size()) / maxProp;
                }
            }
#endif

#if 1


            double tMin = *ctx.root->state_.times.begin();
            double tMax = ctx.root->state_.times.back();
            auto rHist = histogram(ctx.root->state_.times, tMin, tMax);
            auto cHist = histogram(child.state_.times, tMin, tMax);

#if 1
            {
                std::stringstream ss;
                for (const auto &e : rHist) {
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
#endif

            // find the smallest rHist bin the child has historically contributed to
            int64_t smallest = -1;
            {
                int64_t cnt = std::numeric_limits<int64_t>::max();
                for (size_t i = 0; i < rHist.size(); ++i) {
                    if (rHist[i] > 0 && rHist[i] < cnt && cHist[i] > 0) {
                        smallest = i;
                        cnt = rHist[i];
                    }   
                }
            }
            // largest non-zero rHist bin
            int64_t largest = -1;
            {
                int64_t cnt = -1;
                for (size_t i = 0; i < rHist.size(); ++i) {
                    if (rHist[i] > 0 && int64_t(rHist[i]) > cnt) {
                        largest = i;
                        cnt = rHist[i];
                    }   
                }
            }
            // how far that bin is from the max
            double need;
            if (-1 == smallest) {
                need = 0;
            } else {
                need = 1.0 - double(rHist[smallest]) / rHist[largest];
            }

            // TODO: weight by what proportion of the child runs appear in that bin


            STDERR(smallest << " " << largest << " " << need);
            return need;
            // return need * double(cHist[smallest]) / child.state_.times.size(); // makes things worse?
#endif
        }
    }

    static void backprop(Context &ctx, MyNode &node, const Schedule::BenchResult &br) {
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