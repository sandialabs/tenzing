#pragma once

#include <memory>
#include <vector>
#include <map>

#include "cuda_runtime.h"
#include "operation.hpp"

template <typename T>
class Graph {
public:

    typedef std::shared_ptr<T> node_t;
    node_t start_;

    Graph(node_t start) : start_(start) {}


    node_t start() const { return start_; }


    /* create a graph with clone()'ed nodes, except
       src in this graph is replaced with dst in the result graph
    */
    Graph<T> clone_but_replace(node_t dst, node_t src) {



        // clone all nodes, maintain a mapping from original to new
        std::map<node_t, node_t> clones;

        {
            std::vector<node_t> worklist;
            worklist.push_back(start_);
            while(!worklist.empty()) {
                node_t n = worklist.back();
                worklist.pop_back();
                if (clones.count(n)) {
                    continue;
                } else {
                    if (src == n) {
                        clones[n] = dst;
                    } else {
                        clones[n] = n->clone();
                    }
                    for (node_t s : n->succs) {
                        worklist.push_back(s);
                    }
                }
            }
        }

        // connect the new nodes in the same way as the old nodes
        for (auto &kv : clones) {
            node_t o = kv.first; // original
            node_t c = kv.second; // clone

            for (node_t os : o->succs) {
                c->succs.insert(clones[os]);
            }
            for (node_t op : o->preds) {
                c->preds.insert(clones[op]);
            }
        }


        // return the new graph
        Graph<T> ret;
        ret.start_ = clones[start_];
        return ret;
    }


    


};





/* turn a graph that may have Nodes that are not CpuNodes into
   possible 
*/
std::vector<Graph<Node>> use_streams(const Graph<Node> &orig, const std::vector<cudaStream_t> &streams);