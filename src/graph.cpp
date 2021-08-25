#include "sched/graph.hpp"
#include "sched/macro_at.hpp"

#include <mpi.h>

#include <sstream>
#include <fstream>

template<>
void Graph<Node>::dump_graphviz(const std::string &path) const {

    std::ofstream os(path);
    os << "digraph D {";

    // dump nodes
    for (const auto &kv : succs_) {
        os << "node_" << kv.first.get() << " [label=\"";
        os << kv.first->name();

        if (auto ss = std::dynamic_pointer_cast<StreamedOp>(kv.first)) {
            os << "\nstream " << ss->stream();
        }

        os << "\"];\n";
    }

    // dump edges
    for (const auto &kv : succs_) {
        for (const auto &succ : kv.second) {
            os << "node_" << kv.first.get() << " -> " << "node_" << succ.get() << "\n";
        }
    }

    os << "}\n";
}

std::vector<Graph<Node>> use_streams(const Graph<Node> &orig, const std::vector<cudaStream_t> &streams) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    using node_t = std::shared_ptr<Node>;
    using gpu_t = std::shared_ptr<GpuNode>;

    std::vector<Graph<Node>> graphlist;
    std::vector<Graph<Node>> ret;

    graphlist.push_back(orig);

    while (!graphlist.empty()) {
        if (0 == rank) {
            std::cerr << "graphlist.size() = " << graphlist.size() << "\n";
        }

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

/*
The assignment strategy follows

The dependence of operations does not matter, it's just the combinatorial assignment of operations to resources.
For example, two resource types: Streams S0,S1 and CPUs C0, C1, C2

Crawl the graph to extract a list of Stream Operations SO and CPU operations CO
SO: [O0, O1, O3] 
CO: [O2, O4]

Then the possible stream operations are

   O0 O1 O3
   --------
0. S0 S0 S0
1. S0 S0 S1
2. S0 S1 S0
3. S0 S1 S1
4. S1 S0 S0 (no, same as 3.)
5. S1 S0 S1 (no, same as 2.)
...

In short to generate all the unique assignments:
* 0th operation can be assiged to 1st resource
* 1st operation can be assigned to 1st,2nd resource
* 2nd operation can be assigned to 1st,2nd,3rd resource
* 3rd operation ... 1st...4th resource

Of course, then you need to have cartesian product of resource type assignments as well

Here, we only have one resource type (streams)
*/
std::vector<Graph<Node>> use_streams2(const Graph<Node> &orig, const std::vector<cudaStream_t> &streams) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    using node_t = std::shared_ptr<Node>;
    using gpu_t = std::shared_ptr<GpuNode>;

    // extract all GPU operations
    std::vector<gpu_t> gpuOps;

    for (auto &kv : orig.succs_) {
        node_t n = kv.first;
        if (gpu_t gpu = std::dynamic_pointer_cast<GpuNode>(n)) {
            gpuOps.push_back(gpu);
        }
    }

    // each assignment is a vector of which resource each gpuOp is assigned to
    std::vector<std::vector<int>> assignments;

    size_t numAssignments = 1;
    for (size_t i = 0; i < gpuOps.size(); ++i) {
        numAssignments *= std::min(i+1, streams.size());
    }

    std::cerr << "creating " << numAssignments << " assignments for " << gpuOps.size() << " operations in " << streams.size() << " streams\n";

    for (size_t ai = 0; ai < numAssignments; ++ai) {

        std::vector<int> assignment;
        int div = numAssignments;
        for (size_t gi = 0; gi < gpuOps.size(); ++gi) {
            int numOptions = std::min(gi+1, streams.size());
            div /= numOptions;

            // std::cerr << "ass " << ai << ": idx=" << gi << " div=" << div << " mod=" << numOptions << "\n";

            assignment.push_back((ai / div) % numOptions);
        }
        assignments.push_back(assignment);
    }

    std::vector<Graph<Node>> ret;

    for (const auto &assignment : assignments) {
        // for (auto &e : assignment) {
        //     std::cerr << " " << e;
        // }
        // std::cerr << "\n";

        // get a copy of the graph with all the same nodes
        Graph<Node> ng(orig);

        // replace each GPU node with a streamedNode
        for (size_t ai = 0; ai < assignment.size(); ++ai) {
            gpu_t gpu = gpuOps[ai];
            auto copy = std::shared_ptr<GpuNode>(static_cast<GpuNode*>(gpu->clone().release()));
            if (!copy) THROW_RUNTIME("should have been a gpu node");

            size_t si = assignment[ai];
            if (si >= streams.size()) THROW_RUNTIME("stream index too large");
            cudaStream_t stream = streams[si];
            auto streamed = std::make_shared<StreamedOp>(copy, stream);
            ng.replace(gpu, streamed);
        }


        ret.push_back(ng);
    }

    return ret;
}


bool is_equivalent_stream_mapping(const Graph<Node> &a, const Graph<Node> &b) {

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::map<cudaStream_t, cudaStream_t> bij; // stream b for stream a

    // if a's stream matches b's under bijection, or new bijection entry,
    // return true. else return false.
    auto check_or_update_bijection = 
    [&](const std::shared_ptr<Node> &_a, const std::shared_ptr<Node> &_b) -> bool {

        auto aa = std::dynamic_pointer_cast<StreamedOp>(_a);
        auto bb = std::dynamic_pointer_cast<StreamedOp>(_b);
        if (aa && bb) {
            if (bij.count(aa->stream()) && bb->stream() != bij[aa->stream()])
            {
                return false;
            }
            if (bij.count(bb->stream()) && aa->stream() != bij[bb->stream()])
            {
                return false;
            }
            bij[aa->stream()] = bb->stream();
            bij[bb->stream()] = aa->stream();
        }
        return true;
    };

    // same number of operations in the two graphs
    if (a.preds_.size() != b.preds_.size()) {
        return false;
    }

#if 0
    if ( 0 == rank ) {
        std::cerr <<  "a\tb:\n";
        auto ai = a.succs_.begin();
        auto bi = b.succs_.begin();
        for(;ai != a.succs_.end() && bi != b.succs_.end(); ++ai, ++bi) {

            std::cerr << ai->first->name() << "\t" << bi->first->name() << "\n";

        }
        std::cerr <<  "\n";
    }
#endif

    // we're guaranteed consistent operation sorting
    auto ai = a.succs_.begin();
    auto bi = b.succs_.begin();
    for(;ai != a.succs_.end() && bi != b.succs_.end(); ++ai, ++bi) {

        const auto u_a = ai->first;
        const auto u_b = bi->first;

#if 0
        if (0 == rank) std::cerr << "compare " << u_a->name() << " vs. " << u_b->name() << "\n";
#endif 
        if (!u_a->eq(u_b)) { // not same operation
            // if (0 == rank) std::cerr << "FALSE: unequal operations: " << u_a->name() << " vs. " << u_b->name() << "\n";
            return false;
        }

        // check if operations are equivalent under stream bijection
        if (!check_or_update_bijection(u_a, u_b)) {
            // if (0 == rank) std::cerr << "FALSE: failed bijection\n";
            return false;
        }

        // same number of successors
        if (a.succs_.at(u_a).size() != b.succs_.at(u_b).size()) {
            // if (0 == rank) std::cerr << "FALSE: different number of successors\n";
            return false;
        }
        // same number of predecessors
        if (a.preds_.at(u_a).size() != b.preds_.at(u_b).size()) {
            // if (0 == rank) std::cerr << "FALSE: different number of predecessors\n";
            return false;
        }

        // all succs must be equal. no need to check bijection since we
        // check each node's equality under bijection later
        {
            const auto &as = a.succs_.at(u_a);
            const auto &bs = b.succs_.at(u_b);

            auto asi = as.begin();
            auto bsi = bs.begin();

            for (; asi != as.end() && bsi != bs.end(); ++asi, ++bsi) {
                if (!((*asi)->eq(*bsi))) {
                    // if (0 == rank) std::cerr << "FALSE: succ mismatch\n";
                    return false;
                }
            }
        }

        // all preds must be equal
        {
            const auto &as = a.preds_.at(u_a);
            const auto &bs = b.preds_.at(u_b);

            auto asi = as.begin();
            auto bsi = bs.begin();

            for (; asi != as.end() && bsi != bs.end(); ++asi, ++bsi) {
                if (!((*asi)->eq(*bsi))) {
                    // if (0 == rank) std::cerr << "FALSE: pred mismatch\n";
                    return false;
                }
            }
        }
    }
    return true;

}


Graph<Node> insert_synchronization(Graph<Node> &orig) {

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

                            /*
                            there may already be a StreamWait that is an immediate pred of v that causes v to wait on u.
                            if so, use that. otherwise, make one
                            */
                            node_t w;
                            for (auto &p : orig.preds_[v]) {
                                if (auto n = std::dynamic_pointer_cast<StreamWait>(p)) {
                                    if (n->waiter() == ug->stream() && n->waitee() == vg->stream()) {
                                        w = n;
                                        break;
                                    }
                                }
                            }
                            if (!w) w = std::make_shared<StreamWait>(vg->stream(), ug->stream());

                            // add u -> w -> v
                            orig.then(u, w);
                            orig.then(w, v);

                            // remove u->v
                            orig.succs_[u].erase(v);
                            orig.preds_[v].erase(u);

                            // update w's name
                            {
                                auto ssw = std::dynamic_pointer_cast<StreamWait>(w);
                                ssw->update_name(orig.preds_[w], orig.succs_[w]);
                            }

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

                        // if (0 == rank) {
                        //     std::cerr << "need " << ug->name() << " -> ss -> " << vc->name() << "\n";
                        // }

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
                        if (!w) {
                            w = std::make_shared<StreamSync>(ug->stream());
                        } else {
                            // if (0 == rank) std::cerr << "using " << w->name() << " (already existed)\n";
                        }

                        // add u -> w -> v
                        orig.then(u, w);
                        orig.then(w, v);

                        // remove u->v
                        orig.succs_[u].erase(v);
                        orig.preds_[v].erase(u);

                        // update w's name
                        {
                            auto ssw = std::dynamic_pointer_cast<StreamSync>(w);
                            ssw->update_name(orig.preds_[w], orig.succs_[w]);
                        }

                        // if (0 == rank) std::cerr << "ss is called " << w->name() << "\n";

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