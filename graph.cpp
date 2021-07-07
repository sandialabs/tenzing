#include "graph.hpp"

#include "at.hpp"



std::vector<Graph<Node>> use_streams(const Graph<Node> &orig, const std::vector<cudaStream_t> &streams) {

    using node_t = std::shared_ptr<Node>;
    using gpu_t = std::shared_ptr<GpuNode>;

    std::vector<Graph<Node>> graphlist;
    std::vector<Graph<Node>> ret;

    graphlist.push_back(orig);

    while (!graphlist.empty()) {

        // work from the back of the list.
        Graph<Node> g = graphlist.back();
        graphlist.pop_back();

        // find a GpuNode in the graph
        bool hasGpuNode = false;
        for (auto &kv : g.succs_) {
            node_t n = kv.first;
            if (gpu_t gpu = std::dynamic_pointer_cast<GpuNode>(n)) {
                
                // create a copy of that graph, with the GPU node replaced by a StreamedNode for each stream
                for (cudaStream_t stream : streams) {

                    // get a copy of the gpu node. we know it's a GPU node so cast away
                    auto copy = std::shared_ptr<GpuNode>(static_cast<GpuNode*>(gpu->clone().release()));
                    if (!copy) throw std::runtime_error(AT);

                    auto streamed = std::make_shared<StreamedOp>(copy, stream);
                    Graph<Node> ng = g.clone_but_replace(streamed, gpu);
                    graphlist.push_back(ng);
                }
                hasGpuNode = true;
                break; // GpuNode found, and new graphs have been added. stop searching
            }
        }

        // if no Gpu node in the graph, nowhere to apply streams
        if (!hasGpuNode) {
            ret.push_back(g);
        }
    }

    return ret;
}


Graph<Node> insert_synchronization(Graph<Node> &orig) {

    using node_t = Graph<Node>::node_t;

    bool changed = true;
    while(changed) {
        changedloop:
        changed = false;
        for (auto &kv : orig.succs_) {
            node_t u = kv.first;
            for (node_t v : kv.second) {

                // gpu -> gpu needs synchronization if in different streams
                {
                    auto ug = std::dynamic_pointer_cast<StreamedOp>(u);
                    auto vg = std::dynamic_pointer_cast<StreamedOp>(v);
                    if (ug && vg) {
                        if (ug->stream() != vg->stream()) {
                            node_t w = std::make_shared<StreamWait>(vg->stream(), ug->stream());

                            // add u -> w -> v
                            orig.then(u, w);
                            orig.then(w, v);

                            // remove u->v
                            orig.succs_[u].erase(v);
                            orig.preds_[v].erase(u);

                            changed = true;
                            goto changedloop;
                        }
                    }

                }
                    

                // gpu -> cpu non-gpu and non-sync needs synchronization
                /*
                a special case: two gpu ops on the same stream have a shared cpu successor
                we clean this up later
                */
                {
                    auto ug = std::dynamic_pointer_cast<StreamedOp>(u);
                    auto vc = std::dynamic_pointer_cast<CpuNode>(v);
                    auto vss = std::dynamic_pointer_cast<StreamSync>(v);
                    auto vsw = std::dynamic_pointer_cast<StreamWait>(v);
                    auto vso = std::dynamic_pointer_cast<StreamedOp>(v);
                    if (ug && vc && !vss && !vsw && !vso) {

                        /* a pred of v that syncs u's stream may already exist.
                           if not, make one one */
                        node_t w;
                        for (auto &p : orig.preds_[v]) {
                            if (auto n = std::dynamic_pointer_cast<StreamSync>(p)) {
                                if (n->stream() == ug->stream()) {
                                    w = n;
                                    break;
                                }
                            }
                        }
                        if (!w) w = std::make_shared<StreamSync>(ug->stream());

                        // add u -> w -> v
                        orig.then(u, w);
                        orig.then(w, v);

                        // remove u->v
                        orig.succs_[u].erase(v);
                        orig.preds_[v].erase(u);

                        changed = true;
                        goto changedloop;
                    }
                }

                // cpu -> gpu needs nothing 
            }

        }
    }

    return orig;
}