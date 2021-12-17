#pragma once

#include "schedule.hpp"
#include "benchmarker.hpp"

#include <limits>
#include <algorithm>

namespace mcts {


template <typename Strategy>
struct Node {

    using Context = typename Strategy::Context;
    using State = typename Strategy::State;

    Node *parent_;
    std::vector<Node> children_;
    std::shared_ptr<BoundOp> op_;
    bool expanded_;
    bool fullyVisited_; // if this subtree fully expanded
    float valueEstimate_; // an estimate of this node's value if it doesn't have enough playouts
    size_t n_; // # of playouts

    // state required for whatever the strategy is
    State state_;

    Node(const std::shared_ptr<BoundOp> &op) 
        : parent_(nullptr), op_(op), expanded_(false), fullyVisited_(false),
        valueEstimate_(std::numeric_limits<float>::infinity()), // estimate an infinite value before a child is visited
        n_(0) {}

    // how many nodes are in this subtree
    size_t size() const;

    // select successive child nodes until a leaf L is reached
    // a leaf is a node that has a child from which no simulation has been played
    Node &select(Context &ctx, const Graph<OpBase> &g);

    // create unexpanded children for this node
    Node &expand(const Context &ctx, Platform &plat, const Graph<OpBase> &g);

    // true if node can't have any children
    bool is_terminal() const;

    // true if node has unvisited children
    bool is_leaf() const;

    // Get a random rollout from this node
    std::vector<std::shared_ptr<BoundOp>> get_simulation_order(
        Platform &plat,
        const Graph<OpBase> &g
    );

    // backpropagate results up the tree.
    // invokes Strategy::backprop
    void backprop(Context &ctx, const Benchmark::Result &br);

    // ensure that a particular path is available from this node
    Node &expand_order(
        const Context &ctx,
        Platform &plat,
        const Graph<OpBase> &g,
        const std::vector<std::shared_ptr<BoundOp>> &order
    );

    // return the first ancestor that does not have a parent
    const Node &root() const;
    Node &root();

private:
    // ensure that this node's children exist, if there are any
    void ensure_children(const Context &ctx, Platform &plat, const Graph<OpBase> &g);
};



/* return the frontier of nodes from g given already-traversed nodes
   FIXME: this function returns syncs for unsynced grpah nodes, which
    means that multiples syncs for different versions of the same graph
    node can be added to the path.

    it will be replaced with (1),(2),(3) below
*/
std::vector<std::shared_ptr<BoundOp>> get_frontier(
    Platform &plat,
    const Graph<OpBase> &g, 
    const std::vector<std::shared_ptr<BoundOp>> &completed
);

/* 
(1)
  return a frontier of nodes from the graph, with possible platform bindings
*/
std::vector<std::shared_ptr<BoundOp>> get_graph_frontier(
    Platform &plat,
    const Graph<OpBase> &g, 
    const std::vector<std::shared_ptr<BoundOp>> &completed
);

/*
(2)
return a copy of g with an unbound version of op replaced with op
*/
Graph<OpBase> bind_unbound_vertex(
    const Graph<OpBase> &g, 
    std::shared_ptr<BoundOp> &op
);

/*
(3)
considering the `sequence` so far, the graph, and the platform,
return all synchronizations that are needed before op can actually be
appended to the sequence
*/
std::vector<std::shared_ptr<BoundOp>> get_syncs_before_op(
    Platform &plat,
    const Graph<OpBase> &g,
    const std::vector<std::shared_ptr<BoundOp>> &sequence,
    std::shared_ptr<BoundOp> &op
);

template<typename Strategy>
bool Node<Strategy>::is_terminal() const {
    return bool(std::dynamic_pointer_cast<End>(op_));
}

template<typename Strategy>
bool Node<Strategy>::is_leaf() const {
    if (children_.empty()) {
        return true;
    }
    for (const auto &child : children_) {
        if (0 == child.n_) {
            return true;
        }
    }
    return false;
}

/* number of nodes in the subtree (including this one)
*/
template <typename Strategy>
size_t Node<Strategy>::size() const {
    size_t acc = 1; // this node
    for (const auto &child : children_) {
        acc += child.size();
    }
    return acc;
}


/* select successive child nodes until a leaf is reached
   a leaf is a node that has a child from which no simulation has been played

   a terminal node has no possible children (this must be `end` for us?)
*/
template <typename Strategy>
Node<Strategy> &Node<Strategy>::select(Context &ctx, const Graph<OpBase> &g) {

    if (is_leaf() || is_terminal()) {
        return *this;
    } else {

        STDERR(ctx);

        // there should always be children except for terminal nodes
        if (children_.empty()) {
            THROW_RUNTIME("select on " << op_->desc() << " but children are empty!");
        }

        // ubc of each child
        std::vector<float> ucts;
        for (const auto &child : children_) {

            // value of child
            const float exploit = Strategy::select(ctx, *this, child);
            const float c = std::sqrt(2.0f);

            // value of exploring this child
            float explore;

            if (child.fullyVisited_) { // penalize children with no more orderings to visit
                explore = -std::numeric_limits<float>::infinity();
            } else if (1 > child.n_) {
                explore = c * std::sqrt(std::log(n_)/1);
            } else {
                explore = c * std::sqrt(std::log(n_)/child.n_);
            }

            if (std::isnan(explore)) {
                STDERR("explore value was NaN");
            }
            if (std::isnan(exploit)) {
                STDERR("exploit score was NaN");
            }

            const float uct = exploit + explore;

            STDERR(
            child.op_->desc() 
            << ": n=" << child.n_ 
            << " explore=" << explore 
            << " exploit=" << exploit
            << " state=" << child.state_
            );
            ucts.push_back(uct);

        }

        // argmax(ucts)
        // if it's a tie, return a random one since the children
        // are in no particular order
        size_t im = 0;
        {
            float m = -std::numeric_limits<float>::infinity();
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
void Node<Strategy>::backprop(Context &ctx, const Benchmark::Result &br) {
    ++n_; // additional playout

    if (children_.empty()) {
        if (expanded_) {
            fullyVisited_ = expanded_;
            STDERR(op_->name() << " fully visisted (no children)");
        }
    } else { // if all children are visited
        bool allChildrenVisited = true;
        for (Node &child : children_) {
            allChildrenVisited = allChildrenVisited && child.fullyVisited_;
        }
        if (allChildrenVisited) {
            fullyVisited_ = true;
            STDERR(op_->name() << " fully visisted (all children explored)");
        }
    }


    Strategy::backprop(ctx, *this, br);
    if (parent_) {
        parent_->backprop(ctx, br);
    }
}


template <typename Strategy>
Node<Strategy> &Node<Strategy>::expand_order(
    const Context &ctx, 
    Platform &plat,
    const Graph<OpBase> &g, 
    const std::vector<std::shared_ptr<BoundOp>> &order
) {

    // expansion done
    if (order.empty()) {
        return *this;
    }

    // the first node in the order may be this node, since the order will start with the root
    if (*order.begin() == op_) {
        std::vector<std::shared_ptr<BoundOp>> rest(order.begin() + 1, order.end());
        return expand_order(ctx, plat, g, rest);
    }

    // now the first operation in the order should be the child after this one that the order
    // descends to

    STDERR("ensure " << op_->desc() << "'s children exist...");
    ensure_children(ctx, plat, g); // make sure this node's children exist
    
    const std::shared_ptr<BoundOp> &head = *order.begin();
    STDERR("looking for " << head->desc() << " among children...");

    for (auto &node : children_) {
        STDERR("...comparing with " << node.op_->desc());
        if (node.op_->eq(head)) { // == doesn't work here because each created BoundGpuOp is different
            std::vector<std::shared_ptr<BoundOp>> rest(order.begin() + 1, order.end());
            return node.expand_order(ctx, plat, g, rest);
        }
    }
    THROW_RUNTIME("couldn't find " << head->desc() << ", expected as child of " << op_->desc());
}



template <typename Strategy>
Node<Strategy> &Node<Strategy>::expand(const Context &ctx, Platform &plat, const Graph<OpBase> &g) {

    STDERR("ensure_children...");
    ensure_children(ctx, plat, g);
    
    // chose a child node to return
    if (children_.empty()) {
        return *this; // terminal
    } else {

        // TODO: don't automatically prioritize unplayed nodes
        // give them the value of their parents and use UCTS

        // first unplayed node
        for (auto &child : children_) {
            if (0 == child.n_) {
                return child;
            }
        }
        THROW_RUNTIME("expand called on non-leaf node (has children, but no unplayed children)");
    }
}




template <typename Strategy>
std::vector<std::shared_ptr<BoundOp>> Node<Strategy>::get_simulation_order(Platform &plat, const Graph<OpBase> &g) {

    // get the path we took to be here
    std::vector<std::shared_ptr<BoundOp>> path;
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

    
    STDERR("random path...");
    while(true) {
        std::vector<std::shared_ptr<BoundOp>> frontier = get_frontier(plat, g, path);
        if (frontier.empty()) {
            break;
        }
        size_t ii = rand() % frontier.size();
        auto op = frontier[ii];
        path.push_back(op);
    }

    {
        std::string s;
        for (const auto &op : path) {
            s += op->desc();
            s += ", ";
        }
        STDERR("get_simulation_order result is: " << s);
    }

    return path;
}

template<typename Strategy>
const Node<Strategy> &Node<Strategy>::root() const {
    if (!parent_) {
        return *this;
    } else {
        return parent_->root();
    }
}

template<typename Strategy>
Node<Strategy> &Node<Strategy>::root() {
    if (!parent_) {
        return *this;
    } else {
        return parent_->root();
    }
}



template <typename Strategy>
void Node<Strategy>::ensure_children(const Context &, Platform &plat, const Graph<OpBase> &g) {

    if (expanded_) {
        return;
    }

    // get the path we took to be here
    std::vector<std::shared_ptr<BoundOp>> path;
    Node *current = this;
    while(current) {
        path.push_back(current->op_);
        current = current->parent_;
    }
    std::reverse(path.begin(), path.end());

    STDERR("get_frontier() ...");
    std::vector<std::shared_ptr<BoundOp>> frontier = get_frontier(plat, g, path);

    // create child nodes in frontier
    STDERR("create child nodes...");
    for (const std::shared_ptr<BoundOp> &op : frontier) {
        Node node(op);
        node.parent_ = this;
        children_.push_back(node);
    }
    STDERR("created " << children_.size() << " children");

    // mark node expanded
    expanded_ = true;
}




} // namespace mcts