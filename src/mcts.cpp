#include "sched/mcts.hpp"

#include <limits>
#include <algorithm>

/*
  challenges with MCTS
  if the stream assignment is considered jointly with ordering, some parent-child
  pairs will need syncs between, and some will not (can't just attach sync to parent)

  what is a "win" and a "loss"
  does win/loss affect how results will be found?

*/

namespace mcts {

struct Context {
    std::vector<int> streamAssignment;
};

struct Node {

    Node *parent_;
    std::vector<Node> children_;
    std::shared_ptr<CpuNode> op_;

    std::vector<double> times_; // the times of evaluations of this node

    bool expanded_;

    Node(const std::shared_ptr<CpuNode> &op) : parent_(nullptr), op_(op), expanded_(false) {}

    // true if node can't have children
    bool is_terminal(const Graph<CpuNode> &g);
    bool is_leaf() const {return children_.empty(); }

    // select successive child nodes until a leaf L is reached
    Node &select(const Graph<CpuNode> &g);

    // create unexanded children for this node
    Node &expand(const Graph<CpuNode> &g);

    // Measure a random ordering from this node
    void simulate(const Graph<CpuNode> &g);

    void backprop();
};

/* select successive child nodes until a leaf is reached
*/
Node &Node::select(const Graph<CpuNode> &g) {
    if (is_leaf()) {
        return *this;
    } else {
        // TODO: guided selection
        return children_[rand() % children_.size()].select(g);
    }
}

Node &Node::expand(const Graph<CpuNode> &g) {

    typedef std::shared_ptr<CpuNode> Op;

    if (expanded_) {
        return *this;
    }
    // can't use is_terminal here since it calls expand

    // get the path we took to be here
    std::vector<Op> path;
    Node *current = this;
    while(current) {
        path.push_back(current->op_);
        current = current->parent_;
    }

    // create a child for each successor in the path
    std::vector<Op> frontier;
    for (const auto &op : path ) {
        for (const auto &child : g.succs_.at(op)) {
            frontier.push_back(child);
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

    // create all child nodes
    for (const Op &op : frontier) {
        Node node(op);
        node.parent_ = this;
        children_.push_back(node);
    }

    // mark node expanded
    expanded_ = true;

    // chose a child node to return
    if (children_.empty()) {
        return *this; // terminal
    } else {
        #warning multi-rank seed
        return children_[rand() % children_.size()];
    }
}

void Node::simulate(const Graph<CpuNode> &g) {
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
    {
        for (const auto &op : path ) {
            for (const auto &child : g.succs_.at(op)) {
                frontier.push_back(child);
            }
        }
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
    }


    while(!frontier.empty()) {
        // choose a random node. erase from frontier
        #warning rank ordering
        auto it = frontier.begin() + (rand() % frontier.size());
        frontier.erase(it);

        // add any of its successors not in path or in frontier to frontier
        for (const auto &succ : g.succs_.at(*it)) {
            auto ipath = std::find(path.begin(), path.end(), *it);
            auto ifront = std::find(frontier.begin(), frontier.end(), *it);
            if (path.end() == ipath && frontier.end() == ifront) {
                frontier.push_back(succ);
            }
        }

        // add to path
        path.push_back(*it);
    }

    {
        std::string s;
        for (const auto &op : path) {
            s += op->name();
            s += ", ";
        }
        STDERR("random path is: " << s);
    }

    // benchmark the path
    #warning skeleton
    STDERR("fake benchmark...");
    times_.push_back(rand() % 10);

}

bool Node::is_terminal(const Graph<CpuNode> &g) {
    expand(g);
    return children_.empty();
}

void Node::backprop() {
    times_.clear();

    // update my score from my children
    for (Node &child : children_) {
        for (const auto &time : child.times_) {
            times_.push_back(time);
        }
    }

    // tell my parent to do the same
    if (parent_) {
        parent_->backprop();
    }
}


void mcts(const Graph<CpuNode> &g) {
    Context ctx;

    STDERR("create root...");
    Node root(g.start_);

    for (int i = 0; i < 5; ++i) {
        STDERR("select...");
        Node &selected = root.select(g);
        STDERR("selected " << selected.op_->name());

        STDERR("expand...");
        Node &child = selected.expand(g);
        STDERR("expanded to " << child.op_->name());

        STDERR("simulate...");
        child.simulate(g);

        STDERR("backprop...");
        child.backprop();
    }

}

} // namespace mcts