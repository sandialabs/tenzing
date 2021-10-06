#pragma once

#include "graph.hpp"
#include "schedule.hpp"
#include "trap.hpp"
#include "numeric.hpp"

#include "mpi.h"

#include <vector>
#include <algorithm>
#include <functional>
#include <fstream>
#include <chrono>

/*
  challenges with MCTS
  if the stream assignment is considered jointly with ordering, some parent-child
  pairs will need syncs between, and some will not (can't just attach sync to parent)

  what is a "win" and a "loss"
  does win/loss affect how results will be found?

*/
namespace mcts {

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;

struct SimResult {
    std::vector<std::shared_ptr<CpuNode>> path; // path that is simulated
    Schedule::BenchResult benchResult; // times from the simulation
};

struct Result {
    std::vector<SimResult> simResults;
};

/* options for MCTS
*/
struct Opts {
    size_t nIters; // how many searches to do
    size_t dumpTreeEvery; // how often to dump the tree
    std::string dumpTreePrefix; // prefix to use for the tree
    BenchOpts benchOpts; // options for the benchmark runs

    Opts() : dumpTreeEvery(0) {}
};

void mpi_bcast(std::vector<std::shared_ptr<CpuNode>> &order, MPI_Comm comm);

template <typename Strategy>
struct Node {

    using Context = typename Strategy::Context;

    Node *parent_;
    std::vector<Node> children_;
    std::shared_ptr<CpuNode> op_;

    std::vector<double> times_;

    bool expanded_;

    Node(const std::shared_ptr<CpuNode> &op) : parent_(nullptr), op_(op), expanded_(false) {}

    // true if node can't have children
    bool is_terminal(const Graph<CpuNode> &g);
    bool is_leaf() const {
        if (children_.empty()) {
            return true;
        }
        for (const auto &child : children_) {
            if (child.times_.empty()) {
                return true;
            }
        }
        return false;
    }

    // select successive child nodes until a leaf L is reached
    Node &select(Context &ctx, const Graph<CpuNode> &g);

    // create unexanded children for this node
    Node &expand(const Context &ctx, const Graph<CpuNode> &g);

    // Measure a random ordering from this node
    std::vector<std::shared_ptr<CpuNode>> get_simulation_order(const Graph<CpuNode> &g);
    SimResult simulate(const Graph<CpuNode> &g);
};

/* base class for strategy context.
   provides a no-op ostream output overload
*/
struct StrategyContext {
};
inline std::ostream &operator<<(std::ostream &os, const StrategyContext &) {return os;}

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

    // score child
    static double select(const Context &ctx, const MyNode &/*parent*/, const MyNode &child) {
        double v;
        if (child.times_.empty()) {
            v =  0;
        } else {
            double acc = child.times_[child.times_.size() * 0 / 100];
            v = (acc - ctx.minT) / (ctx.maxT - ctx.minT);
            v = 1-v;
            if (v < 0) v = 0;
            if (v > 1) v = 1;
        }
        return v;
    }

    static void backprop(Context &ctx, MyNode &node, const Schedule::BenchResult &br) {

        double elapsed = br.pct10;
        node.times_.push_back(elapsed);

        // order times smallest to largest
        std::sort(node.times_.begin(), node.times_.end());

        // tell my parent to do the same
        if (node.parent_) {
            backprop(ctx, *node.parent_, br);
        }
        // keep track of a window of central values to compare speeds against
        else {
            size_t loi = node.times_.size() * 0 / 100;
            size_t hii = node.times_.size() * 100 / 100 - 1;
            ctx.minT = node.times_[loi];
            ctx.maxT = node.times_[hii];
        }
    }
};



/* score child higher if it is anticorrelated with parent
*/
struct AntiCorrelation {

    using MyNode = Node<AntiCorrelation>;

    struct Context : public StrategyContext {}; // unused

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
        if (parent.times_.size() < 2 || child.times_.size() < 2) {
            v = 0;
        } else {

            double tMin = std::min(*parent.times_.begin(), *child.times_.begin());
            double tMax = std::max(parent.times_.back(), child.times_.back());

            // score children by inverse correlation with parent
            auto pHist = histogram(parent.times_, tMin, tMax);
            auto cHist = histogram(child.times_, tMin, tMax);
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

    static void backprop(Context &ctx, MyNode &node, const Schedule::BenchResult &br) {
        double elapsed = br.pct10;
        node.times_.push_back(elapsed);

        // order times smallest to largest
        std::sort(node.times_.begin(), node.times_.end());

        // tell my parent to do the same
        if (node.parent_) {
            backprop(ctx, *node.parent_, br);
        }
    }
};

/* score child higher if it is anticorrelated with parent
*/
struct NormalizedAntiCorrelation {

    using MyNode = Node<NormalizedAntiCorrelation>;

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
                c = 2 - c; // [0,2] anticorrelation
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
            c = 2 - c; // [0,2] anticorrelation
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


/* select successive child nodes until a leaf is reached
   a leaf is a node that has a child from which no simulation has been played
*/
template <typename Strategy>
Node<Strategy> &Node<Strategy>::select(Context &ctx, const Graph<CpuNode> &g) {
    if (is_leaf() || is_terminal(g)) {
        return *this;
    } else {

        STDERR(ctx);

        // ubc of each child
        std::vector<float> ucts;
        for (const auto &child : children_) {

            // value of child
            float v;
#if 0
            // assign a value proportional to how much of the
            //space between the slowest and fastest run this child represents
            if (child.times_.size() < 2) {
                // none or 1 measurement doesn't tell anything
                // about how fast this program is relative
                // to the overall
                v = 0;
                // prefer children that do not have enough runs yet
                // v = std::numeric_limits<double>::infinity();
            } else {
                double tMax = child.times_[child.times_.size() * 98 / 100];
                double tMin = child.times_[child.times_.size() * 2 / 100];
                v = (tMax - tMin) / (ctx.maxT - ctx.minT);
                v = 300.0 * stddev(child.times_) / avg(child.times_);
                // if (v < 0) v = 0;
                // if (v > 1) v = 1;
            }
#endif
            // compute the average speed of the child and compare to the window
            v = Strategy::select(ctx, *this, child);

            const float c = 1.41; // sqrt(2)
            const float n = times_.size();
            const float nj = child.times_.size();

            if (nj > 0) {
                double explore = c * std::sqrt(std::log(n)/nj);
                double exploit = v;
                double uct = exploit + explore;
                STDERR(
                child.op_->name() 
                << ": n=" << child.times_.size() 
                << " explore=" << explore 
                << " exploit=" << exploit
                << " minT=" << *child.times_.begin()
                << " maxT=" << child.times_.back()
                );
                ucts.push_back(uct);
            } else {
                ucts.push_back(std::numeric_limits<float>::infinity());
            }
        }

        // argmax(ucts)
        // if it's a tie, return a random one since the children
        // are in no particular order
        size_t im = 0;
        {
            float m = -1 * std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < ucts.size(); ++i) {
                if (ucts[i] > m) {
                    m = ucts[i];
                    im = i;
                }
            }
            std::vector<size_t> choices;
            for (size_t i = 0; i < ucts.size(); ++i) {
                if (ucts[i] == m) {
                    choices.push_back(i);
                }
            }
            im = choices[rand() % choices.size()];

            STDERR("selected " << children_[im].op_->name() << " uct=" << m);
        }


        return children_[im].select(ctx, g);
    }
}

template <typename Strategy>
Node<Strategy> &Node<Strategy>::expand(const Context &, const Graph<CpuNode> &g) {
    typedef std::shared_ptr<CpuNode> Op;

    if (is_terminal(g)) {
        return *this;
    }

    // create child nodes if needed
    if (!expanded_) {
        // get the path we took to be here
        std::vector<Op> path;
        Node *current = this;
        while(current) {
            path.push_back(current->op_);
            current = current->parent_;
        }

        // make sure each successor of every node in the path appears
        // exactly once in the frontier list
        std::vector<Op> frontier;
        for (const auto &op : path ) {
            for (const auto &child : g.succs_.at(op)) {
                if (frontier.end() == std::find(frontier.begin(), frontier.end(), child)) {
                    frontier.push_back(child);
                }
            }
        }

        // remove all ops in the frontier that we've already done
        for (const Op &op : path) {
            while (true) {
                auto it = std::find(frontier.begin(), frontier.end(), op);
                if (it != frontier.end()) {
                    frontier.erase(it);
                } else {
                    break;
                }
            }
        }

        // remove all ops in the frontier that have a predecessor that's not
        // in the path
        {
            bool changed = true;
            while(changed) {
                changed = false;
                for (auto fi = frontier.begin(); fi != frontier.end(); ++fi) {
                    for (const auto &pred : g.preds_.at(*fi)) {
                        // predecessor is not in the path
                        if (path.end() == std::find(path.begin(), path.end(), pred)) {
                            frontier.erase(fi);
                            changed = true;
                            goto endloop;
                        }
                    }
                }
                endloop: ;
            }
        }

        // create all child nodes
        for (const Op &op : frontier) {
            Node node(op);
            node.parent_ = this;
            children_.push_back(node);
        }
        STDERR("expanded " << children_.size() << " children");

        // mark node expanded
        expanded_ = true;
    }

    // chose a child node to return
    if (children_.empty()) {
        return *this; // terminal
    } else {
        // random
        // return children_[rand() % children_.size()];

        // first unplayed node
        for (auto &child : children_) {
            if (child.times_.empty()) {
                return child;
            }
        }
        // if all children have been played, this is not a leaf node
        THROW_RUNTIME("unreachable");
    }
}


template <typename Strategy>
SimResult Node<Strategy>::simulate(const Graph<CpuNode> &g) {
    typedef std::shared_ptr<CpuNode> Op;

    // get the path we took to be here
    std::vector<Op> path = get_simulation_order(g);

    // benchmark the path
    STDERR("single-rank benchmark...");

    SimResult result;
    result.benchResult = Schedule::benchmark(path, MPI_COMM_WORLD);
    result.path = path;
    return result;
}


template <typename Strategy>
std::vector<std::shared_ptr<CpuNode>> Node<Strategy>::get_simulation_order(const Graph<CpuNode> &g) {
    typedef std::shared_ptr<CpuNode> Op;

    // get the path we took to be here
    std::vector<Op> path;
    Node *current = this;
    while(current) {
        path.push_back(current->op_);
        current = current->parent_;
    }
    std::reverse(path.begin(), path.end());
    {
        std::string s;
        for (const auto &op : path) {
            s += op->name();
            s += ", ";
        }
        STDERR("path is: " << s);
    }


    // choose a random traversal of the remaining nodes
    std::vector<Op> frontier;
    // add all successors in the path that have not already been visited
    // and have all predecessors complete
    {
        // add one copy of each successor to all nodes in the path
        for (const auto &op : path ) {
            for (const auto &child : g.succs_.at(op)) {

                bool unique = (frontier.end() == std::find(frontier.begin(), frontier.end(), child));
                bool notDone = (path.end() == std::find(path.begin(), path.end(), child));
                bool predsDone = true;
                for (const auto &pred : g.preds_.at(child)) {
                    if (path.end() == std::find(path.begin(), path.end(), pred)) {
                        predsDone = false;
                        break;
                    }
                }
                // STDERR(child->name() << " " << unique << " " << notDone << " " << predsDone);
                if (unique && notDone && predsDone) {
                    frontier.push_back(child);
                }
            }
        }
    }

    STDERR("random path...");
    while(!frontier.empty()) {
        // choose a random node that's up next
        size_t ii = rand() % frontier.size();
        auto op = frontier[ii];
    
        // add to path
        path.push_back(op);
        // STDERR("current path is ...");
        // for (auto & e : path) {
        //     STDERR(e->name());
        // }

        // add its successors if they're not in the path, they haven't been done,
        // and its preds are done
        for (const auto &succ : g.succs_.at(op)) {
            bool unique = (frontier.end() == std::find(frontier.begin(), frontier.end(), succ));
            bool notDone = (path.end() == std::find(path.begin(), path.end(), succ));
            bool predsDone = true;
            for (const auto &pred : g.preds_.at(succ)) {
                if (path.end() == std::find(path.begin(), path.end(), pred)) {
                    predsDone = false;
                    break;
                }
            }
            // STDERR(succ->name() << " " << unique << " " << notDone << " " << predsDone);
            if (unique && notDone && predsDone) {
                frontier.push_back(succ);
            }
        }

        // erase from frontier
        frontier.erase(frontier.begin() + ii);
    }

    {
        std::string s;
        for (const auto &op : path) {
            s += op->name();
            s += ", ";
        }
        STDERR("random path is: " << s);
    }

    return path;
}


template<typename Strategy>
bool Node<Strategy>::is_terminal(const Graph<CpuNode> &g) {
    return g.succs_.at(op_).empty();
}

template<typename Strategy>
void dump_graphviz(const std::string &path, const Node<Strategy> &root) {

    using Node = Node<Strategy>;

    STDERR("write " << path);
    std::ofstream os(path);

    std::function<void(const Node &)> dump_nodes = [&](const Node &node) -> void {
        os << "node_" << &node << " [label=\"";
        os << node.op_->name();
        os << "\n" << node.times_.size();
        if (!node.times_.empty()) {
            os << "\n" << node.times_[0];
            os << "\n" << node.times_.back();
        }
        os << "\"];\n";

        for (const auto &child : node.children_) {
            if (!child.times_.empty()) {
                dump_nodes(child);
            }
        }
    };

    std::function<void(const Node &)> dump_edges = [&](const Node &node) -> void {
        for (const Node &child : node.children_) {
            if (!child.times_.empty()) {
                os << "node_" << &node << " -> " << "node_" << &child << "\n";
            }
        }
        for (const auto &child : node.children_) {
            if (!child.times_.empty()) {
                dump_edges(child);
            }
        }
    };

    os << "digraph D {";
    dump_nodes(root);
    dump_edges(root);
    os << "}\n";
}

template <typename Strategy>
Result mcts(const Graph<CpuNode> &g, MPI_Comm comm, const Opts &opts = Opts()) {

    using Context = typename Strategy::Context;
    using Node = Node<Strategy>;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // warm up MPI
    {
        std::vector<char> bytes(size);
        MPI_Alltoall(MPI_IN_PLACE, 1, MPI_BYTE, bytes.data(), 1, MPI_BYTE, comm);
    }

    STDERR("create root...");
    Node root(g.start_);

    Result result;

    // print results so far if interrupted
    std::function<void(int)> printResults = [&result](int sig) -> void {
        (void) sig;
        for (const auto &simres : result.simResults) {
            std::cout << simres.benchResult.pct10 << ",";
            for (const auto &op : simres.path) {
                std::cout << op->name() << ",";
            }
            std::cout << "\n"; 
        }
    };
    if (0 == rank) {
        register_handler(printResults);
    }

    Context ctx;

    // get a list of all nodes in the graph

    for (size_t iter = 0; iter < opts.nIters; ++iter) {

        // need to do two simulations of expansion to estimate value
        // initialize list with all nodes in g
        std::vector<std::shared_ptr<CpuNode>> order1;
        for (const auto &kv : g.succs_) {
            order1.push_back(kv.first);
        }
        // auto order2 = order1;
        

        Node *child = nullptr; // result of expansion step
        if (0 == rank) {
            STDERR("select...");
            Node &selected = root.select(ctx, g);
            STDERR("selected " << selected.op_->name());

            STDERR("expand...");
            child = &selected.expand(ctx, g);
            STDERR("expanded to " << child->op_->name());

            STDERR("simulate...");
            order1 = child->get_simulation_order(g);
            // order2 = child->get_simulation_order(g);

            STDERR("remove extra syncs...");
            Schedule::remove_redundant_syncs(order1);
            // Schedule::remove_redundant_syncs(order2);
        }

        // distributed order to benchmark to all ranks
        
        if ( 0 == rank ) STDERR("distribute order...");
        mpi_bcast(order1, comm);

        // benchmark the order
        {
            // warmup
            // BenchOpts opts;
            // Schedule::benchmark(order, comm, opts);
        }
        MPI_Barrier(comm);
        if ( 0 == rank ) STDERR("benchmark...");
        Schedule::BenchResult benchResult1 =
            Schedule::benchmark(order1, comm, opts.benchOpts);
        
        MPI_Barrier(comm);
        if (0 == rank) {
            SimResult simres;
            simres.path = order1;
            simres.benchResult = benchResult1;
            result.simResults.push_back(simres);

            STDERR("backprop...");
            Strategy::backprop(ctx, *child, benchResult1);
        }

        if (0 == rank && opts.dumpTreeEvery != 0 && iter % opts.dumpTreeEvery == 0) {
            std::string treePath = "mcts_";
            treePath += std::to_string(iter);
            treePath += ".dot";
            dump_graphviz(treePath, root);
        }

    }

    unregister_handler();
    return result;


}

} // namespace mcts