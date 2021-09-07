#include "sched/mcts.hpp"

#include <limits>

/*
  challenges with MCTS
  if the stream assignment is considered jointly with ordering, some parent-child
  pairs will need syncs between, and some will not (can't just attach sync to parent)

  what is a "win" and a "loss"
  does win/loss affect how results will be found?

*/

namespace mcts {

struct Context {
    std::vector<Node *> path;
};

struct Node {
    typedef std::shared_ptr<::Node> Op;

    Op op_;
    Node *parent_;
    std::vector<Op> children_;
    Graph<::Node> overlay_; // changes to G relevant to this node (added synchronization, etc)

    bool expanded_;

    Node() : parent_(nullptr), expanded_(false), 
        minTime(std::numeric_limits<double>::infinity()),
        maxTime(-std::numeric_limits<double>::infinity()) {}
    Node(const Op &op) : Node() {
        op_ = op;
    }

    // create unexanded children for this node
    void expand(Context &ctx, const Graph<CpuNode> &g);
    void update_score();

    double minTime;
    double maxTime;
};



void Node::expand(Context &ctx, const Graph<CpuNode> &g) {

    // if this node is followed by a CPU 

    // create a child for each successor in the graph
    for (const auto &op : g.succs_.at(op_)) {
        children_.push_back(Node(op));
    }

    // remove all children that have an operation that is an acestor of this node
    for (const Op &op : ctx.path) {
        for (auto ni = children_.begin(), ni != children_.end(); ++ni) {

            // child is exactly a match for an ancestor
            if (ni->op_ == op) {
                children_.erase(ni);
                break;
            }
            // child is a StreamedOp of an ancestor
            {
                auto so = std::dynamic_pointer_cast<StreamedOp>(*(ni->op_));
                if (so) {
                    if (os->node_ == op) {
                        children_.erase(ni);
                        break;
                    }
                }
            }


        }
    }

    // any added children that are a GPU node that will (when expanded)
    // be followed by a CPU node need to be followed by a cudaEventRecord
    // expanding the CER should have the possible options 

    // some children may need to be replaced with a streamed version

    // when 

    // any GPU->CPU needs a cudaEventRecord immediately after the GPU

    // any GPU->GPU may need a cudaEventRecord immediate after


    // mark node expanded
    expanded_ = true;
}

void Node::update_score() {
    // update my score from my children
    for (Node &child : children_) {
        minTime = std::min(minTime, child.minTime);
        maxTime = std::max(maxTime, child.maxTime);
    }
    
    // tell my parent to do the same
    parent->update_score();
}

// do MCTS for node
void mcts_helper(Context &ctx, Node &node, const Graph<CpuNode> &g) {

    // add this node to the descent context
    ctx.path.push_back(&node);

    if (!node.expanded_) {
        node.expand(ctx, g);
    }

    // node has no children
    if (node.children_.empty()) {
        // this is a complete implementation, evaluate it
        double time = 0;
        node.minTime = std::min(node.minTime, time);
        node.maxTime = std::max(node.maxTime, time);

        // propogate results up to parents
        parent_->update_score();
    } else { // choose a child to descend down
        size_t selection = rand() % children_.size();
        mctx_helper(ctx, children_[selection], g);
    }
}

Node root;

void mcts(Graph<CpuNode> &g) {

    if (!root.op_) {
        root.op_ = g.start();
    }

    Context ctx;

    mcts_helper(ctx, root, g);

}

} // namespace mcts