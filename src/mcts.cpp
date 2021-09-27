#include "sched/mcts.hpp"

#include "sched/schedule.hpp"

#include <limits>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;

/*
  challenges with MCTS
  if the stream assignment is considered jointly with ordering, some parent-child
  pairs will need syncs between, and some will not (can't just attach sync to parent)

  what is a "win" and a "loss"
  does win/loss affect how results will be found?

*/

namespace mcts {

struct Context {
    double minT;
    double maxT;
};


void mpi_bcast(std::vector<std::shared_ptr<CpuNode>> &order, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // bcast number of operations in order
    int orderSize = order.size();
    MPI_Bcast(&orderSize, 1, MPI_INT, 0, comm);

    // bcast length of name of each operation
    std::vector<int> nameLengths;
    for (const auto &op : order) {
        nameLengths.push_back(op->name().size());
    }
    MPI_Bcast(nameLengths.data(), nameLengths.size(), MPI_INT, 0, comm);

    // bcast names
    size_t totalLength = 0;
    for (auto l : nameLengths) {
        totalLength += l;
    }
    std::vector<char> allNames;
    if (0 == rank) {
        for (const auto &op : order) {
            for (auto c : op->name()) {
                allNames.push_back(c);
            }
        }
    } else {
        allNames.resize(totalLength);
    }
    MPI_Bcast(allNames.data(), allNames.size(), MPI_CHAR, 0, comm);

    // break into strings
    std::vector<std::string> names;
    size_t off = 0;
    for (auto l : nameLengths) {
        names.push_back(std::string(&allNames[off], l));
        off += l;
    }

    // find corresponding op in order if recieving
    if (0 != rank) {
        std::vector<size_t> pos;
        for (const std::string &name : names) {

            bool found = false;
            for (size_t oi = 0; oi < order.size(); ++oi) {
                if (order[oi]->name() == name) {
                    pos.push_back(oi);
                    found = true;
                    break;
                }
            }
            if (!found) {
                THROW_RUNTIME("couldn't find op for name " << name);
            }
        }

        // reorder operations
        std::vector<std::shared_ptr<CpuNode>> perm;
        for (size_t oi : pos) {
            perm.push_back(order[oi]);
        }
        order = perm;
    }
}



struct Node {

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
    Node &select(const Context &ctx, const Graph<CpuNode> &g);

    // create unexanded children for this node
    Node &expand(const Context &ctx, const Graph<CpuNode> &g);

    // Measure a random ordering from this node
    std::vector<std::shared_ptr<CpuNode>> get_simulation_order(const Graph<CpuNode> &g);
    SimResult simulate(const Graph<CpuNode> &g);

    // Get a plan to send to other nodes


    void backprop(double med);
};

/* select successive child nodes until a leaf is reached
   a leaf is a node that has a child from which no simulation has been played
*/
Node &Node::select(const Context &ctx, const Graph<CpuNode> &g) {
    if (is_leaf() || is_terminal(g)) {
        return *this;
    } else {
        // ubc of each child
        std::vector<float> ucts;
        for (const auto &child : children_) {
            // value of child
            // assign a value proportional to how much of the
            //space between the slowest and fastest run this child represents
            float v;
            if (child.times_.size() < 2) {
                // none or 1 measurement doesn't tell anything
                // about how fast this program is relative
                // to the overall
                v = 0.5;
            } else {
                double tMax = child.times_.back();
                double tMin = child.times_[0];
                v = (tMax - tMin) / (ctx.maxT - ctx.minT);
                if (v < 0) v = 0;
                if (v > 1) v = 1;
            }
            const float c = 1.41; // sqrt(2)
            const float n = times_.size();
            const float nj = child.times_.size();

            if (nj > 0) {
                double explore = c * std::sqrt(std::log(n)/nj);
                double exploit = v;
                double uct = exploit + explore;
                STDERR(child.op_->name() << ": explore=" << explore << " exploit=" << exploit);
                ucts.push_back(uct);
            } else {
                ucts.push_back(std::numeric_limits<float>::infinity());
            }
        }

        // argmax(ucts)
        size_t im = 0;
        {
            float m = -1 * std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < ucts.size(); ++i) {
                if (ucts[i] > m) {
                    m = ucts[i];
                    im = i;
                }
            }
            STDERR("selected " << children_[im].op_->name() << " uct=" << m);
        }


        return children_[im].select(ctx, g);
    }
}

Node &Node::expand(const Context &, const Graph<CpuNode> &g) {
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



SimResult Node::simulate(const Graph<CpuNode> &g) {
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


std::vector<std::shared_ptr<CpuNode>> Node::get_simulation_order(const Graph<CpuNode> &g) {
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



bool Node::is_terminal(const Graph<CpuNode> &g) {
    return g.succs_.at(op_).empty();
}

void dump_graphviz(const std::string &path, const Node &root) {
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
            dump_nodes(child);
        }
    };

    std::function<void(const Node &)> dump_edges = [&](const Node &node) -> void {
        for (const Node &child : node.children_) {
            os << "node_" << &node << " -> " << "node_" << &child << "\n";
        }
        for (const auto &child : node.children_) {
            dump_edges(child);
        }
    };

    os << "digraph D {";
    dump_nodes(root);
    dump_edges(root);
    os << "}\n";
}

void Node::backprop(double med) {
    // median
    times_.push_back(med);

    // order times smallest to largest
    std::sort(times_.begin(), times_.end());

    // tell my parent to do the same
    if (parent_) {
        parent_->backprop(med);
    }
}




/* only rank 0 will do the MCTS.
   it will send the ordering to test to the other ranks
*/
Result mcts(const Graph<CpuNode> &g, MPI_Comm comm, const Opts &opts) {

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

    Context ctx;
    ctx.minT =  std::numeric_limits<double>::infinity();
    ctx.maxT = -std::numeric_limits<double>::infinity();

    // get a list of all nodes in the graph

    for (int i = 0; i < 10; ++i) {

        // initialize list with all nodes in g
        std::vector<std::shared_ptr<CpuNode>> order;
        for (const auto &kv : g.succs_) {
            order.push_back(kv.first);
        }

        Node *child = nullptr; // result of expansion step
        if (0 == rank) {
            STDERR("select...");
            Node &selected = root.select(ctx, g);
            STDERR("selected " << selected.op_->name());

            STDERR("expand...");
            child = &selected.expand(ctx, g);
            STDERR("expanded to " << child->op_->name());

            STDERR("simulate...");
            order = child->get_simulation_order(g);

            STDERR("remove extra syncs...");
            Schedule::remove_redundant_syncs(order);
        }

        // distributed order to benchmark to all ranks
        
        if ( 0 == rank ) STDERR("distribute order...");
        mpi_bcast(order, comm);

        // benchmark the order
        {
            // warmup
            // BenchOpts opts;
            // Schedule::benchmark(order, comm, opts);
        }
        MPI_Barrier(comm);
        if ( 0 == rank ) STDERR("benchmark...");
        Schedule::BenchResult benchResult = Schedule::benchmark(order, comm, opts.benchOpts);
        
        MPI_Barrier(comm);
        if ( 0 == rank ) STDERR("backprop...");
        if (0 == rank) {
            SimResult simres;
            simres.path = order;
            simres.benchResult = benchResult;
            result.simResults.push_back(simres);

            if (benchResult.pct10 < ctx.minT) {
                ctx.minT = benchResult.pct10;
            }
            if (benchResult.pct10 > ctx.maxT) {
                ctx.maxT = benchResult.pct10;
            }

            STDERR("backprop...");
            child->backprop(benchResult.pct10);
        }

        if (0 == rank && i % opts.dumpTreeEvery == 0) {
            std::string treePath = "mcts_";
            treePath += std::to_string(i);
            treePath += ".dot";
            dump_graphviz(treePath, root);
        }

    }

    for (const auto &simres : result.simResults) {
        std::cout << simres.benchResult.pct10 << ",";
        for (const auto &op : simres.path) {
            std::cout << op->name() << ",";
        }
        std::cout << "\n";
        
    }

    return result;
}




} // namespace mcts