#pragma once

#include <memory>
#include <vector>
#include <map>
#include <set>

#include "sched/cuda_runtime.h"
#include "sched/operation.hpp"
#include "sched/ops_cuda.hpp"
#include "sched/macro_at.hpp"

template <typename T>
class Graph {
public:

    typedef std::shared_ptr<T> op_t;
    typedef std::set<op_t, OpBase::compare_lt> OpSet;
    op_t start_;

    /* successors and predecessors of each node */
    typedef std::map<op_t, OpSet, OpBase::compare_lt> OpMap;
    OpMap succs_;
    OpMap preds_;

    Graph() = default;
    Graph(op_t start) : start_(start) {
        succs_[start] = {};
        preds_[start] = {};
    }

    // add a and b to the graph, if they're not present, and an edge a->b. return b
    const op_t &then (const op_t &a, const op_t &b) {
        succs_[a].insert(b);
        succs_[b]; // ensure b exists, but we have no info about successors

        preds_[a]; // a exists, but no info about predecessors
        preds_[b].insert(a);
        return b;
    }

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

    // op_t &start() { return start_; }
    const op_t &start() const { return start_; }


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
    Graph<T> clone_but_replace(op_t dst, op_t src) {

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
        Graph<T> ret;
        ret.start_ = clones[start_];

        // connect the new nodes in the same way as the old nodes
        for (auto &kv : clones) {
            op_t o = kv.first; // original
            op_t c = kv.second; // clone
            for (op_t os : succs_[o]) {
                ret.then(c, clones[os]);
            }
        }

        // return the new graph
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
        Graph<T> ret;
        ret.start_ = clones[start_];

        // connect the new nodes in the same way as the old nodes
        for (auto &kv : clones) {
            op_t &o = kv.first; // original
            op_t &c = kv.second; // clone
            for (op_t &os : succs_.at(o)) {
                ret.then(c, clones[os]);
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

    template<typename U>
    Graph<U> nodes_cast() const {
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


    OpSet &preds(T* tp) {
        for (auto &kv : preds_) {
            if (kv.first.get() == tp) {
                return kv.second;
            }
        }
        throw std::runtime_error(AT);
    }

    typename OpMap::const_iterator preds_find(T* tp) const {
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

    OpSet &succs(T* tp) {
        for (auto &kv : succs_) {
            if (kv.first.get() == tp) {
                return kv.second;
            }
        }
        throw std::runtime_error(AT);
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

    typename OpMap::const_iterator succs_find_or_find_unbound(const op_t &key) const {
        typename OpMap::const_iterator it = succs_.find(key);
        if (succs_.end() == it) {
            if (auto bound = std::dynamic_pointer_cast<BoundGpuOp>(key)) {
                it = succs_.find(bound->unbound());
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

    void dump_graphviz(const std::string &path) const;
};





/* turn a graph that has GpuNodes into all possible combinations that only have CpuNodes
*/
std::vector<Graph<OpBase>> use_streams(const Graph<OpBase> &orig, const std::vector<cudaStream_t> &streams);
std::vector<Graph<OpBase>> use_streams2(const Graph<OpBase> &orig, const std::vector<cudaStream_t> &streams);

/* insert required synchronizations between GPU-GPU and CPU-CPU nodes
*/
Graph<OpBase> insert_synchronization(Graph<OpBase> &orig);

/* returns true if a and b are the same under a stream bijection

    every node u_a should have a corresponding u_b with a consistent mapping
    i.e. u_b.stream = map[u_a.stream]

    "corresponding" means u_a.eq(ub) and u_a's preds/succs eq u_b's preds/succs 
*/
bool is_equivalent_stream_mapping(const Graph<OpBase> &a, const Graph<OpBase> &b);