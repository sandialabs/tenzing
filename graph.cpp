#include "graph.hpp"



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
        std::vector<node_t> nodelist;
        std::set<node_t> visited;
        nodelist.push_back(g.start());
        while(!nodelist.empty()) {
            node_t n = nodelist.back();
            nodelist.pop_back();
            if (visited.count(n)) {
                continue;
            } else {
                for (node_t s : n->succs) {
                    nodelist.push_back(s);
                }
                visited.insert(n);
            }

            // if there is a GPU node in the graph
            if (gpu_t gpu = std::dynamic_pointer_cast<GpuNode>(n)) {
                
                // create a copy of that graph, with the GPU node replaced by a StreamedNode for each stream
                for (cudaStream_t stream : streams) {

                    // get a copy of the gpu node. we know it's a GPU node so cast away
                    auto copy = std::unique_ptr<GpuNode>(static_cast<GpuNode*>(gpu->clone().release()));

                    auto streamed = std::make_shared<StreamedOp>(std::move(copy), stream);
                    Graph<Node> ng = g.clone_but_replace(streamed, gpu);
                    graphlist.push_back(ng);
                }
                hasGpuNode = true;
                break; // GpuNode found, and new graphs have been added. stop searching
            }
        }
        if (!hasGpuNode) {
            ret.push_back(g);
        }
    }

    return ret;
}