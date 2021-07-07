#pragma once

#include <memory>
#include <vector>
#include <map>
#include <set>

#include "cuda_runtime.h"
#include "operation.hpp"

template <typename T>
class Graph {
public:

    typedef std::shared_ptr<T> node_t;
    node_t start_;

    /* successors and predecessors of each node */
    std::map<node_t, std::set<node_t>> succs_;
    std::map<node_t, std::set<node_t>> preds_;

    Graph() = default;
    Graph(node_t start) : start_(start) {
        succs_[start] = {};
        preds_[start] = {};
    }

    // add a and b to the graph, if they're not present, and an edge a->b. return b
    std::shared_ptr<Node> then (std::shared_ptr<Node>a, std::shared_ptr<Node>b) {
        succs_[a].insert(b);
        succs_[b]; // ensure b exists, but we have no info about successors

        preds_[a]; // a exists, but no info about predecessors
        preds_[b].insert(a);
        return b;
    }

    void dump_helper(node_t u, node_t v) {
        std::cerr << u->name() << " -> " << v->name() << "\n";
        for (node_t s : succs_[v]) {
            dump_helper(v, s);
        }
    }

    void dump() {
        for (node_t s : succs_[start_]) {
            dump_helper(start_, s);
        }
    }

    node_t start() { return start_; }


    /* create a graph with clone()'ed nodes, except
       src in this graph is replaced with dst in the result graph
    */
    Graph<T> clone_but_replace(node_t dst, node_t src) {



        // clone all nodes, maintain a mapping from original to new
        std::map<node_t, node_t> clones;

        {
            for (auto &kv : succs_) {
                if (src == kv.first) {
                    clones[kv.first] = dst;
                } else {
                    clones[kv.first] = kv.first->clone();
                }
            }
        }

        // create edges in the new graph
        Graph<T> ret;
        ret.start_ = clones[start_];

        // connect the new nodes in the same way as the old nodes
        for (auto &kv : clones) {
            node_t o = kv.first; // original
            node_t c = kv.second; // clone
            for (node_t os : succs_[o]) {
                ret.then(c, clones[os]);
            }
        }

        // return the new graph
        return ret;
    }


    


};





/* turn a graph that has GpuNodes into all possible combinations that only have CpuNodes
*/
std::vector<Graph<Node>> use_streams(const Graph<Node> &orig, const std::vector<cudaStream_t> &streams);

/* insert required synchronizations between GPU-GPU and CPU-CPU nodes
*/
Graph<Node> insert_synchronization(Graph<Node> &orig);