/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include "benchmarker.hpp"
#include "schedule.hpp"
#include "sequence.hpp"
#include "decision.hpp"
#include "state.hpp"

#include <martinmoene/optional.hpp>

#include <algorithm>
#include <limits>

namespace mcts {

/* since rollout may or may not expand, later backprop needs to know
   which node to start backprop from
*/

template <typename Strategy> struct Node {

  template <typename T>
  using Optional = nonstd::optional<T>;

  struct RolloutResult {
    Sequence<BoundOp> sequence;
    Node *backpropStart;
  };

  using Context = typename Strategy::Context;
  using State = typename Strategy::State;

  Node *parent_;
  std::vector<Node> children_;
  Optional<std::shared_ptr<BoundOp>> op_;
  bool expanded_;
  bool fullyVisited_;   // if this subtree fully expanded
  float valueEstimate_; // an estimate of this node's value if it doesn't have enough playouts
  size_t n_;            // # of playouts

  Graph<OpBase> graph_;

  // state required for whatever the strategy is
  State state_;

  Node(const Graph<OpBase> &graph, const std::shared_ptr<BoundOp> &op) : parent_(nullptr), op_(op), expanded_(false), fullyVisited_(false),
        valueEstimate_(std::numeric_limits<float>::infinity()), // estimate an infinite value before
                                                                // a child is visited
        n_(0), graph_(graph) {}
  Node(const Graph<OpBase> &graph) : parent_(nullptr), expanded_(false), fullyVisited_(false),
        valueEstimate_(std::numeric_limits<float>::infinity()), n_(0), graph_(graph) {}
  Node() : Node(Graph<OpBase>()) {}

  // subtree size (including this one)
  size_t size() const;               // how many nodes
  size_t unvisited_size() const;     // how nodes without a rollout
  size_t fully_visited_size() const; // how many fully-visited nodes

  // select successive child nodes until a leaf L is reached
  // a leaf is a node that has a child from which no simulation has been played
  Node &select(Context &ctx);

  // create unexpanded children for this node
  Node &expand(Platform &plat);

  // true if node can't have any children
  bool is_terminal() const;

  // true if node has unvisited children
  bool is_leaf() const;

  // Get a random rollout from this node
  // optionally expand nodes in the tree along the way
  RolloutResult get_rollout(Platform &plat, bool expand = true);

  // backpropagate results up the tree.
  // invokes Strategy::backprop
  void backprop(Context &ctx, const Benchmark::Result &br);

  // return the first ancestor that does not have a parent
  const Node &root() const;
  Node &root();

  // return one or more lines formatted as a graphviz label
  std::string graphviz_label() const;
  std::string graphviz_name() const;

  // get the sequence through the tree to get here (including this node)
  Sequence<BoundOp> get_sequence() const;

  /// \brief short description of node
  std::string desc() const;

private:
  // create all the children of a node
  std::vector<Node> create_children(Platform &plat, bool quiet = false);

  // create children, and attach to node
  void ensure_children(Platform &plat);
};

/* return the frontier of nodes from g given already-traversed nodes
   FIXME: this function returns syncs for unsynced grpah nodes, which
    means that multiples syncs for different versions of the same graph
    node can be added to the path.

    it will be replaced with (1),(2),(3) below
*/
std::vector<std::shared_ptr<BoundOp>>
get_frontier(Platform &plat, const Graph<OpBase> &g,
             const std::vector<std::shared_ptr<BoundOp>> &completed);

#if 0
/*
(1)
  return a frontier of nodes from the graph, with possible platform bindings
*/
std::vector<std::shared_ptr<BoundOp>>
get_graph_frontier(Platform &plat, const Graph<OpBase> &g,
                   const Sequence<BoundOp> &completed, bool quiet = false);
#endif

/*
(2)
return a copy of g with an unbound version of op replaced with op
*/
Graph<OpBase> bind_unbound_vertex(const Graph<OpBase> &g, const std::shared_ptr<BoundOp> &op);

/*
(3)
considering the `sequence` so far, the graph, and the platform,
return all synchronizations that are needed before op can actually be
appended to the sequence
*/
std::vector<std::shared_ptr<BoundOp>>
get_syncs_before_op(const Graph<OpBase> &g,
                          const Sequence<BoundOp> &completed,
                          const std::shared_ptr<BoundOp> &op);

template <typename Strategy> bool Node<Strategy>::is_terminal() const {

  return op_ && bool(std::dynamic_pointer_cast<End>(*op_));
}

template <typename Strategy> bool Node<Strategy>::is_leaf() const {
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

template <typename Strategy> size_t Node<Strategy>::size() const {
  size_t acc = 1; // this node
  for (const auto &child : children_) {
    acc += child.size();
  }
  return acc;
}

template <typename Strategy> size_t Node<Strategy>::unvisited_size() const {
  size_t acc = 0 == n_ ? 1 : 0; // this node
  for (const auto &child : children_) {
    acc += child.unvisited_size();
  }
  return acc;
}

template <typename Strategy> size_t Node<Strategy>::fully_visited_size() const {
  size_t acc = fullyVisited_ ? 1 : 0; // this node
  for (const auto &child : children_) {
    acc += child.fully_visited_size();
  }
  return acc;
}

/* select successive child nodes until a leaf is reached
   a leaf is a node that has a child from which no simulation has been played

   a terminal node has no possible children (this must be `end` for us?)

   TODO: could do UCTS for all children, and if an unplayed child has the highest value,
   return this node, otherwise, descend.
*/
template <typename Strategy> Node<Strategy> &Node<Strategy>::select(Context &ctx) {
#if 1
  // old way, always return on leaf or terminal
  if (is_leaf() || is_terminal()) {
    return *this;
  } else {

    STDERR(ctx);

    // there should always be children except for terminal nodes
    if (children_.empty()) {
      THROW_RUNTIME("select on " << desc() << " but children are empty!");
    }

    // ubc of each child
    std::vector<float> ucts;
    for (const auto &child : children_) {

      // value of child
      const float exploit = Strategy::select(ctx, child);
      const float c = std::sqrt(2.0f);

      // value of exploring this child
      float explore;

      if (child.fullyVisited_) { // penalize children with no more orderings to visit
        explore = -std::numeric_limits<float>::infinity();
      } else {
        if (0 == child.n_) {
          THROW_RUNTIME("select should return if there is an unplayed child");
        }
        explore = c * std::sqrt(std::log(n_) / child.n_);
      }

      if (std::isnan(explore)) {
        STDERR("explore value was NaN");
      }
      if (std::isnan(exploit)) {
        STDERR("exploit score was NaN");
      }

      const float uct = exploit + explore;

      STDERR(child.desc() << ": n=" << child.n_ << " explore=" << explore
                               << " exploit=" << exploit << " state=" << child.state_);
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

      STDERR("selected " << children_[im].desc() << " uct=" << m);
    }

    return children_[im].select(ctx);
  }
#else

  // new way, do UCTS for all children (even unplayed)
  // if an unplayed child is selected, return this node.
  // otherwise, descent into child

  if (is_terminal() || children_.empty() /*all children unplayed*/) {
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
      const float exploit = Strategy::select(ctx, child);
      if (std::isnan(exploit)) {
        STDERR("exploit score was NaN");
      }
      const float c = std::sqrt(2.0f);

      // value of exploring this child
      float explore;

      if (child.fullyVisited_) { // penalize children with no more orderings to visit
        explore = -std::numeric_limits<float>::infinity();
      } else {
        if (0 == child.n_) {
          explore = c * std::sqrt(std::log(n_) /
                                  0.75); // somewhat more valuable than a node with 1 rollout
        } else {
          explore = c * std::sqrt(std::log(n_) / child.n_);
        }
      }

      if (std::isnan(explore)) {
        STDERR("explore value was NaN");
      }

      const float uct = exploit + explore;

      STDERR(child.op_->desc() << ": n=" << child.n_ << " explore=" << explore
                               << " exploit=" << exploit << " state=" << child.state_);
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

      STDERR("selected " << children_[im].op_->desc() << " uct=" << m);
    }

    if (0 == children_[im].n_) {
      return *this;
    } else {
      return children_[im].select(ctx);
    }
  }
#endif
}

template <typename Strategy>
void Node<Strategy>::backprop(Context &ctx, const Benchmark::Result &br) {
  ++n_; // additional playout

  if (children_.empty()) {
    if (expanded_) {
      fullyVisited_ = expanded_;
      STDERR(desc() << " fully visited (no children)");
    }
  } else { // if all children are visited
    bool allChildrenVisited = true;
    for (Node &child : children_) {
      allChildrenVisited = allChildrenVisited && child.fullyVisited_;
    }
    if (allChildrenVisited) {
      fullyVisited_ = true;
      STDERR(desc() << " fully visited (all children explored)");
    }
  }

  Strategy::backprop(ctx, *this, br);
  if (parent_) {
    parent_->backprop(ctx, br);
  }
}

template <typename Strategy> Node<Strategy> &Node<Strategy>::expand(Platform &plat) {

  STDERR("ensure_children...");
  ensure_children(plat);

  // chose a child node to return
  if (children_.empty()) {
    return *this; // terminal
  } else {
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
typename Node<Strategy>::RolloutResult Node<Strategy>::get_rollout(Platform &plat, bool expand) {

  Node<Strategy>::RolloutResult res;

  // get the path we took to be here, through our parent
  if (parent_) {
    res.sequence = parent_->get_sequence();
  }
  {
    std::string s;
    for (const auto &op : res.sequence) {
      s += op->name();
      s += ", ";
    }
    STDERR("sequence to rollout start is: " << s);
  }

  if (expand) {
    STDERR("get_rollout with expansion");
  } else {
    STDERR("get_rollout");
  }

  // if we don't expand, expand/traverse subtree in a copy of this node that will be
  // discarded when this function returns.
  // Otherwise, just expand/traverse the subtree in this node so it persists in the tree
  Node copy;
  Node *currNode = nullptr;
  if (expand) {
    currNode = this;
  } else {
    copy = *this;
    currNode = &copy;
  }

  while (currNode) {
    // if expanded the tree, backprop from the new leaf.
    if (expand) {
      res.backpropStart = currNode;
    }

    // add current node to path
    if (currNode->op_) res.sequence.push_back(*(currNode->op_));

    // create children
    currNode->ensure_children(plat);

    // select from children at random
    if (currNode->children_.empty()) {
      currNode = nullptr;
    } else {
      currNode = &currNode->children_[rand() % currNode->children_.size()];
    }
  }

  // otherwise, this is the leaf and backprop from here
  if (!expand) {
    res.backpropStart = this;
  }

  {
    std::string s;
    for (const auto &op : res.sequence) {
      s += op->desc();
      s += ", ";
    }
    STDERR("get_rollout result is: " << s);
  }

  if (!res.backpropStart) {
    THROW_RUNTIME("");
  }
  return res;
}

template <typename Strategy> const Node<Strategy> &Node<Strategy>::root() const {
  if (!parent_) {
    return *this;
  } else {
    return parent_->root();
  }
}

template <typename Strategy> Node<Strategy> &Node<Strategy>::root() {
  if (!parent_) {
    return *this;
  } else {
    return parent_->root();
  }
}

template <typename Strategy> std::string Node<Strategy>::graphviz_label() const {

  std::stringstream ss;

  std::shared_ptr<HasStream> s = op_ ? std::dynamic_pointer_cast<HasStream>(*op_) : nullptr;
  std::shared_ptr<HasEvent> e = op_ ? std::dynamic_pointer_cast<HasEvent>(*op_) : nullptr;

  if (s || e) {
    if (s) {
      auto streams = s->get_streams();
      ss << "s ";
      for (auto si = streams.begin(); si < streams.end(); ++si) {
        ss << *si;
        if (si + 1 < streams.end()) {
          ss << ",";
        }
      }
    }
    if (e) {
      if (s)
        ss << " ";
      auto events = e->get_events();
      ss << "e ";
      for (auto ei = events.begin(); ei < events.end(); ++ei) {
        ss << *ei;
        if (ei + 1 < events.end()) {
          ss << ",";
        }
      }
    }
    ss << "\n";
  }

  ss << state_.graphviz_label_line() << "\n";

  std::string str = ss.str();
  while (str.back() == '\n') {
    str.resize(str.size() - 1);
  }
  return str;
}

template <typename Strategy> std::string Node<Strategy>::graphviz_name() const {
  if (op_) {
    return (*op_)->name();
  } else {
    return "non-op decision";
  }
}

template <typename Strategy>
std::vector<Node<Strategy>> Node<Strategy>::create_children(Platform &plat, bool quiet) {
  std::vector<Node<Strategy>> children;

  // get the path we took to be here
  Sequence<BoundOp> path = get_sequence();

  // construct sequential decision state
  SDP::State sdpState(graph_, path);

  // get all possible decisions to make at this state
  std::vector<std::shared_ptr<Decision>> decisions = sdpState.get_decisions(plat);

  // create child nodes in
  for (const auto &decision : decisions) {

    SDP::State cState = sdpState.apply(*decision);

    if (auto eo = std::dynamic_pointer_cast<ExecuteOp>(decision)) {
      children.push_back(Node(cState.graph(), eo->op));
    } else { // otherwise, include just the revised graph
      children.push_back(Node(cState.graph()));
    }

  }

#if 0
  std::vector<std::shared_ptr<BoundOp>> frontier = get_graph_frontier(plat, graph_, path, quiet);
  {
    std::string s;
    for (const auto &op : frontier) {
      s += op->desc();
      s += ", ";
    }
    STDERR("create_children: graph frontier is: " << s);
  }

  /* create child nodes in frontier
     if the child does not require synchronization, use it directly
     if it does, create one child for each possible synchronization
  */
  STDERR("create child nodes...");
  for (const std::shared_ptr<BoundOp> &op : frontier) {
    STDERR("create_children: create child node(s) for " << op->desc());

    // track if the child implies any platform binding
    STDERR("create_children: create graph by replacing unbound with " << op->desc());
    Graph<OpBase> cGraph = bind_unbound_vertex(graph_, op);

    auto syncs = get_syncs_before_op(cGraph, path, op);
    if (!syncs.empty()) {
      STDERR("create_children: " << op->desc() << " required syncs.");
      for (const auto &cSync : syncs) {
        Node node(cSync, cGraph);
        node.parent_ = this;
        children.push_back(node);
      }
    } else {
      Node node(op, cGraph);
      node.parent_ = this;
      children.push_back(node);
    }
  }
  STDERR("created " << children.size() << " children");

  // some children may represent identical sequences
  // e.g., the first two children may assign the same op to two different streams
  // no need to check both subtrees

  for (auto ci = children.begin(); ci < children.end(); ++ci) {
    for (auto cj = ci + 1; cj < children.end(); ++cj) {
      auto seqi = ci->get_sequence();
      auto seqj = cj->get_sequence();
      if (!quiet)
        STDERR("compare " << get_desc_delim(seqi, ",") << " w/ " << get_desc_delim(seqj, ","));
      if (get_equivalence(seqi, seqj)) {
        STDERR("elide duplicate child " << ci->desc() << " (= " << cj->desc() << ") of "
                                        << desc());
        ci = cj = children.erase(ci); // cj will be incremented next loop iter
      }
    }
  }
#endif

  return children;
}

template <typename Strategy> void Node<Strategy>::ensure_children(Platform &plat) {

  if (expanded_) {
    return;
  }
  children_ = create_children(plat, true);
  STDERR("created " << children_.size() << " children");

  // mark node expanded
  expanded_ = true;
}

template <typename Strategy> Sequence<BoundOp> Node<Strategy>::get_sequence() const {
  Sequence<BoundOp> seq;
  const Node *current = this;
  while (current) {
    if (op_) seq.push_back(*(current->op_));
    current = current->parent_;
  }
  std::reverse(seq.begin(), seq.end());
  return seq;
}

template <typename Strategy> std::string Node<Strategy>::desc() const {

  if (op_) {
    return (*op_)->desc();
  } else {
    return "non-op decision";
  }
}

} // namespace mcts