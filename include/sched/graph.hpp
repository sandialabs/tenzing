/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include <map>
#include <memory>
#include <set>
#include <type_traits>
#include <vector>

#include "sched/cast.hpp"
#include "sched/cuda/ops_cuda.hpp"
#include "sched/macro_at.hpp"
#include "sched/operation.hpp"

template <typename T> class Graph {
public:
  typedef std::shared_ptr<T> op_t;
  typedef std::set<op_t, OpBase::compare_lt> OpSet;
  op_t start_;
  op_t finish_;

  /* successors and predecessors of each node */
  typedef std::map<op_t, OpSet, OpBase::compare_lt> OpMap;
  OpMap succs_;
  OpMap preds_;

private:
  /*! \brief create a graph with (start) -> (finish)
   */
  Graph(const std::shared_ptr<Start> &start, const std::shared_ptr<Finish> &finish)
      : start_(start), finish_(finish) {
    then_raw(start_, finish_);
  }

public:
  /*! \brief create a graph with (Start) -> (Finish)
   */
  Graph() : Graph(std::make_shared<Start>(), std::make_shared<Finish>()) {}

  /*! \brief add a and b to the graph, if they're not present, and an edge a->b. return b
   */
  void start_then(const op_t &a) {
    erase_edge_only(start_, finish_);
    then_raw(start_, a);
  }

  /*! \brief add a and b to the graph, if they're not present, and an edge a->b. return b
   */
  void then_finish(const op_t &a) {
    erase_edge_only(start_, finish_);
    then_raw(a, finish_);
  }

  /*! \brief add a and b to the graph, if they're not present, and an edge a->b. return b
   */
  const op_t &then(const op_t &a, const op_t &b) {

    if (std::dynamic_pointer_cast<Start>(a) || std::dynamic_pointer_cast<Start>(b) ||
        std::dynamic_pointer_cast<Finish>(a) || std::dynamic_pointer_cast<Finish>(b)) {
      THROW_RUNTIME("can't insert Start or Finish with then(), use start_then() or then_finish()");
    }

    succs_[a].insert(b);
    succs_[b]; // ensure b exists, but we have no info about successors

    preds_[a]; // a exists, but no info about predecessors
    preds_[b].insert(a);
    return b;
  }

  const op_t &start() const { return start_; }
  const op_t &finish() const { return finish_; }

  /*! \brief all vertices whos only pred is start
   */
  const OpSet start_vertices() const {
    OpSet ret;
    for (const op_t &succ : succs_.at(start_)) {
      const OpSet &preds = preds_.at(succ);
      if (preds.size() == 1 && *preds.begin() == start_) {
        ret.insert(succ);
      }
    }
    return ret;
  }

  /*! \brief all vertices with finish as only successor
   */
  const OpSet finish_vertices() const {
    OpSet ret;
    for (const auto &kv : succs_) {
      if (1 == kv.second.size() && *(kv.second.begin()) == finish_) {
        ret.insert(kv.first);
      }
    }
    return ret;
  }

  size_t vertex_size() const { return succs_.size(); }

  void dump_helper(op_t u, op_t v) {
    std::cerr << u->name() << " -> " << v->name() << "\n";
    for (op_t s : succs_[v]) {
      dump_helper(v, s);
    }
  }

  void dump() {
    for (op_t s : succs_[start_]) {
      dump_helper(start_, s);
    }
  }

  bool contains(const op_t &op) const {
    for (const auto &kv : succs_) {
      if (op->eq(kv.first)) {
        return true;
      }
    }
    return false;
  }

  /* create a graph with clone()'ed nodes, except
     src in this graph is replaced with dst in the result graph
  */
  Graph<T> clone_but_replace(op_t dst, op_t src) const {

    // clone all nodes, maintain a mapping from original to new
    std::map<op_t, op_t, OpBase::compare_lt> clones;
    for (auto &kv : succs_) {
      if (src == kv.first) {
        clones[kv.first] = dst;
      } else {
        clones[kv.first] = kv.first->clone();
      }
    }

    // create edges in the new graph
    Graph<T> ret(SCHED_CAST_OR_THROW(Start, clones[start_]),
                 SCHED_CAST_OR_THROW(Finish, clones[finish_]));
    ret.erase_edge_only(ret.start(), ret.finish());

    // connect the new nodes in the same way as the old nodes
    for (auto &kv : clones) {
      op_t o = kv.first;  // original
      op_t c = kv.second; // clone
      for (op_t os : succs_.at(o)) {
        ret.then_raw(c, clones[os]);
      }
    }

    // return the new graph
    return ret;
  }

  /*! \brief clone the graph with \c op replaced with `graph`
   */
  Graph<T> clone_but_expand(const std::shared_ptr<T> &op, const Graph<OpBase> &graph) const {

    OpSet startSuccs = graph.start_vertices();
    OpSet finishPreds = graph.finish_vertices();

    // clone all nodes, maintain a mapping from original to new
    std::map<op_t, op_t, OpBase::compare_lt> clones;
    for (auto &kv : succs_) {
      clones[kv.first] = kv.first->clone();
    }

    // create edges in the new graph
    Graph<T> ret(SCHED_CAST_OR_THROW(Start, clones[start_]),
                 SCHED_CAST_OR_THROW(Finish, clones[finish_]));
    ret.erase_edge_only(ret.start(), ret.finish());

    for (auto &kv : clones) {
      op_t u = kv.first;   // original
      op_t up = kv.second; // clone

      // all u ->v (up -> vp) edges old (new) graph
      for (op_t v : succs_.at(u)) {
        op_t vp = clones.at(v);

        // op -> v replaced with (preds of end of graph) -> v
        if (u == op) {
          for (const op_t &finish : graph.finish_vertices()) {
            ret.then_raw(finish, vp);
          }
        } else if (v == op) { // u -> op replaced with u -> (succs of start of graph)
          for (const op_t &start : graph.start_vertices()) {
            ret.then_raw(up, start);
          }
        } else {
          ret.then_raw(up, vp);
        }
      }
    }

    return ret;
  }

  /* create a graph with clone()'ed nodes
   */
  Graph<T> clone() const {
    // clone all nodes, maintain a mapping from original to new
    std::map<op_t, op_t, OpBase::compare_lt> clones;
    for (auto &kv : succs_) {
      clones[kv.first] = kv.first->clone();
    }

    // create edges in the new graph
    Graph<T> ret(SCHED_CAST_OR_THROW(Start, clones[start_]),
                 SCHED_CAST_OR_THROW(Finish, clones[finish_]));
    ret.erase_edge_only(ret.start(), ret.finish());

    // connect the new nodes in the same way as the old nodes
    for (auto &kv : clones) {
      const op_t &o = kv.first; // original
      op_t &c = kv.second;      // clone
      for (const op_t &os : succs_.at(o)) {
        ret.then_raw(c, clones[os]);
      }
    }

    return ret;
  }

  /* replace src with dst in this graph
   */
  void replace(op_t src, op_t dst) {

    // maybe replace start
    if (src == start_) {
      start_ = dst;
    }

    OpSet outEdges = succs_.at(src);
    OpSet inEdges = preds_.at(src);

    for (auto &n : outEdges) {
      then(dst, n);
    }
    for (auto &n : inEdges) {
      then(n, dst);
    }

    // remove original node & edges
    erase(src.get());
  }

  template <typename U> Graph<U> nodes_cast() const {
    Graph<U> ret;
    // set start node
    {
      auto s = std::dynamic_pointer_cast<U>(start_);
      if (s) {
        ret.start_ = s;
      } else {
        throw std::runtime_error(AT);
      }
    }

    // recreate edges
    for (auto &kv : succs_) {
      auto u = kv.first;
      for (auto v : kv.second) {
        auto uu = std::dynamic_pointer_cast<U>(u);
        auto vv = std::dynamic_pointer_cast<U>(v);
        if (!uu || !vv) {
          throw std::runtime_error(AT);
        } else {
          ret.then(uu, vv);
        }
      }
    }
    return ret;
  }

  OpSet &preds(T *tp) {
    for (auto &kv : preds_) {
      if (kv.first.get() == tp) {
        return kv.second;
      }
    }
    throw std::runtime_error(AT);
  }

  const OpSet &preds(T *tp) const {
    for (auto &kv : preds_) {
      if (kv.first.get() == tp) {
        return kv.second;
      }
    }
    throw std::runtime_error(AT);
  }

  std::vector<op_t> preds_vec(T *tp) const {
    auto opset = preds(tp);
    std::vector<op_t> ret;
    for (const auto &op : opset) {
      ret.push_back(op);
    }
    return ret;
  }

  typename OpMap::const_iterator preds_find(T *tp) const {
    for (typename OpMap::const_iterator it = preds_.begin(); it != preds_.end(); ++it) {
      if (it->first.get() == tp) {
        return it;
      }
    }
    return preds_.end();
  }

  typename OpMap::const_iterator preds_find_or_find_unbound(const op_t &key) const {
    typename OpMap::const_iterator it = preds_.find(key);
    if (preds_.end() == it) {
      if (auto bound = std::dynamic_pointer_cast<BoundGpuOp>(key)) {
        it = preds_.find(bound->unbound());
      }
    }
    return it;
  }

  OpSet &succs(T *tp) {
    for (auto &kv : succs_) {
      if (kv.first.get() == tp) {
        return kv.second;
      }
    }
    throw std::runtime_error(AT);
  }

  const OpSet &succs(T *tp) const {
    for (auto &kv : succs_) {
      if (kv.first.get() == tp) {
        return kv.second;
      }
    }
    throw std::runtime_error(AT);
  }

  std::vector<op_t> succs_vec(T *tp) const {
    auto opset = succs(tp);
    std::vector<op_t> ret;
    for (const auto &op : opset) {
      ret.push_back(op);
    }
    return ret;
  }

#if 0
    typename OpMap::const_iterator succs_find(T* tp) const {
        for (typename OpMap::const_iterator it = succs_.begin(); it != succs_.end(); ++it) {
            if (it->first.get() == tp) {
                return it;
            }
        }
        return succs_.end();
    }
#endif

  // finds key or key->unbound in succs
  typename OpMap::const_iterator succs_find_or_find_unbound(const op_t &key) const {
    typename OpMap::const_iterator it = succs_.find(key);
    if (succs_.end() == it) {
      if (auto bound = std::dynamic_pointer_cast<BoundGpuOp>(key)) {
        it = succs_.find(bound->unbound());
      }
    }
    return it;
  }

  /// \brief returns v s.t. v->name() == key->name()
  typename OpMap::const_iterator succs_find_name(const std::string &key) const {
    auto it = succs_.begin();
    for (; it != succs_.end(); ++it) {
      if (it->first->name() == key) {
        return it;
      }
    }
    return it;
  }

  void erase(const T *tp) {
  // can't erase start
  if (tp == start_.get()) {
    throw std::runtime_error(AT);
  }

  // erase node's out-edges
  for (auto it = succs_.begin(); it != succs_.end(); ++it) {
    if (it->first.get() == tp) {
      succs_.erase(it);
      break;
    }
  }
  // erase node's in-edges
  for (auto it = preds_.begin(); it != preds_.end(); ++it) {
    if (it->first.get() == tp) {
      preds_.erase(it);
      break;
    }
  }

  // erase edges from other nodes to this node
  for (auto &kv : succs_) {
    for (auto &succ : kv.second) {
      if (tp == succ.get()) {
        kv.second.erase(succ);
        break;
      }
    }
  }

  // erase edges from other nodes to this node
  for (auto &kv : preds_) {
    for (auto &pred : kv.second) {
      if (tp == pred.get()) {
        kv.second.erase(pred);
        break;
      }
    }
  }
}

/*! \brief erase a->b, but leave a, b even if no edges remain
 */
void erase_edge_only(const op_t &a, const op_t &b) {

  // erase succs_[a][b] if exists
  {
    auto ia = succs_.find(a);
    if (succs_.end() != ia) {
      auto ib = ia->second.find(b);
      if (ia->second.end() != ib) {
        ia->second.erase(ib);
      }
    }
  }

  // erase preds_[b][a] if exists
  {
    auto ib = preds_.find(b);
    if (preds_.end() != ib) {
      auto ia = ib->second.find(a);
      if (ib->second.end() != ia) {
        ib->second.erase(ia);
      }
    }
  }
}

/*! \brief return all nodes that have all predecessors in \c visited

    Does not handle any nesting, e.g. the graph has a compound node and one of the choices is in
   visited

   \param visisted the vector of visited predecessors
   \tparam U the type of node in the \c visited vector
*/
template <typename U, typename std::enable_if<std::is_base_of<OpBase, U>::value, bool>::type = true>
std::vector<op_t> frontier(const std::vector<std::shared_ptr<U>> &visited) const {

  STDERR("consider ops with >= 1 pred completed...");
  std::vector<std::shared_ptr<OpBase>> onePredVisited;
  for (const auto &vOp : visited) {
    STDERR("...incl all succs of " << vOp->desc() << " (@" << vOp.get() << ")");

    // some nodes in the path will not be in the graph (inserted syncs)
    // other nodes in the path are bound versions of that in the graph

    auto it = succs_find_or_find_unbound(vOp);
    if (succs_.end() != it) {

      // all successors of a completed op have at least one pred completed
      for (const auto &succ : it->second) {
        // don't add duplicates
        if (onePredVisited.end() == std::find(onePredVisited.begin(), onePredVisited.end(), succ)) {
          onePredVisited.push_back(succ);
        }
      }
    }
  }

  {
    std::stringstream ss;
    ss << "one pred completed: ";
    for (const auto &op : onePredVisited) {
      ss << op->desc() << ",";
    }
    STDERR(ss.str());
  }

  STDERR("reject ops already done or with incomplete preds...");
  std::vector<std::shared_ptr<OpBase>> result;
  for (const auto &vOp : onePredVisited) {
    // reject ops that we've already done
    if (unbound_contains(visited, vOp)) {
      STDERR(vOp->name() << " already done");
      continue;
    }

    // reject ops that all preds are not done
    bool allPredsCompleted = true;
    for (const auto &pred : preds_.at(vOp)) {
      if (!unbound_contains(visited, pred)) {
        STDERR(vOp->name() << " missing pred " << pred->name());
        allPredsCompleted = false;
        break;
      }
    }
    if (!allPredsCompleted) {
      STDERR(vOp->name() << " missing a pred");
      continue;
    }
    result.push_back(vOp);
  }

  return result;
}

void dump_graphviz(const std::string &path) const;

private:
// add a and b to the graph, if they're not present, and an edge a->b. return b
const op_t &then_raw(const op_t &a, const op_t &b) {

  succs_[a].insert(b);
  succs_[b]; // ensure b exists, but we have no info about successors

  preds_[a]; // a exists, but no info about predecessors
  preds_[b].insert(a);
  return b;
}
}
;

/* turn a graph that has GpuNodes into all possible combinations that only have CpuNodes
 */
std::vector<Graph<OpBase>> use_streams(const Graph<OpBase> &orig,
                                       const std::vector<cudaStream_t> &streams);
std::vector<Graph<OpBase>> use_streams2(const Graph<OpBase> &orig,
                                        const std::vector<cudaStream_t> &streams);

/* insert required synchronizations between GPU-GPU and CPU-CPU nodes
 */
Graph<OpBase> insert_synchronization(Graph<OpBase> &orig);

/* returns true if a and b are the same under a stream bijection

    every node u_a should have a corresponding u_b with a consistent mapping
    i.e. u_b.stream = map[u_a.stream]

    "corresponding" means u_a.eq(ub) and u_a's preds/succs eq u_b's preds/succs
*/
bool is_equivalent_stream_mapping(const Graph<OpBase> &a, const Graph<OpBase> &b);

// try to discover an equivalence between two graphs
// if not, return falsy
Equivalence get_equivalence(const Graph<OpBase> &a, const Graph<OpBase> &b);
