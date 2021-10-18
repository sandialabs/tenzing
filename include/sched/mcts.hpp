#pragma once

#include "graph.hpp"
#include "schedule.hpp"
#include "trap.hpp"
#include "numeric.hpp"
#include "mcts_node.hpp"

#include "mpi.h"

#include <vector>
#include <functional>
#include <fstream>
#include <chrono>

/*
  challenges with MCTS
  if the stream assignment is considered jointly with ordering, some parent-child
  pairs will need syncs between, and some will not (can't just attach sync to parent)

  what is a "win" and a "loss"
  does win/loss affect how results will be found?

*/
namespace mcts {

struct SimResult {
    std::vector<std::shared_ptr<CpuNode>> path; // path that is simulated
    Benchmark::Result benchResult; // times from the simulation
};

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;

struct Result {
    std::vector<SimResult> simResults;
};

/* options for MCTS
*/
struct Opts {
    size_t nIters; // how many searches to do
    size_t dumpTreeEvery; // how often to dump the tree
    std::string dumpTreePrefix; // prefix to use for the tree
    BenchOpts benchOpts; // options for the runs

    Opts() : dumpTreeEvery(0) {}
};

void mpi_bcast(std::vector<std::shared_ptr<CpuNode>> &order, MPI_Comm comm);



template<typename Strategy>
void dump_graphviz(const std::string &path, const Node<Strategy> &root) {

    using Node = Node<Strategy>;

    STDERR("write " << path);
    std::ofstream os(path);

    std::function<void(const Node &)> dump_nodes = [&](const Node &node) -> void {
        os << "node_" << &node << " [label=\"";
        os << node.op_->name();
        os << "\n" << node.state_.times.size();
        if (!node.state_.times.empty()) {
            os << "\n" << node.state_.times[0];
            os << "\n" << node.state_.times.back();
        }
        os << "\"];\n";

        for (const auto &child : node.children_) {
            if (!child.state_.times.empty()) {
                dump_nodes(child);
            }
        }
    };

    std::function<void(const Node &)> dump_edges = [&](const Node &node) -> void {
        for (const Node &child : node.children_) {
            if (!child.state_.times.empty()) {
                os << "node_" << &node << " -> " << "node_" << &child << "\n";
            }
        }
        for (const auto &child : node.children_) {
            if (!child.state_.times.empty()) {
                dump_edges(child);
            }
        }
    };

    os << "digraph D {";
    dump_nodes(root);
    dump_edges(root);
    os << "}\n";
}

template <typename Strategy, typename Benchmarker>
Result mcts(const Graph<CpuNode> &g, Benchmarker &benchmarker, MPI_Comm comm, const Opts &opts = Opts()) {

    using Context = typename Strategy::Context;
    using Node = Node<Strategy>;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    STDERR("create root...");
    Node root(g.start_);

    Result result;

    // print results so far if interrupted
    std::function<void(int)> printResults = [&result](int sig) -> void {
        (void) sig;
        for (const auto &simres : result.simResults) {
            std::cout << simres.benchResult.pct10 << ",";
            for (const auto &op : simres.path) {
                std::cout << op->name() << ",";
            }
            std::cout << "\n"; 
        }
    };
    if (0 == rank) {
        register_handler(printResults);
    }

    Context ctx;

    // get a list of all nodes in the graph

    for (size_t iter = 0; iter < opts.nIters; ++iter) {

        // need to do two simulations of expansion to estimate value
        // initialize list with all nodes in g
        std::vector<std::shared_ptr<CpuNode>> order1;
        for (const auto &kv : g.succs_) {
            order1.push_back(kv.first);
        }
        // auto order2 = order1;
        

        Node *child = nullptr; // result of expansion step
        if (0 == rank) {
            STDERR("select...");
            Node &selected = root.select(ctx, g);
            STDERR("selected " << selected.op_->name());

            STDERR("expand...");
            child = &selected.expand(ctx, g);
            STDERR("expanded to " << child->op_->name());

            STDERR("simulate...");
            order1 = child->get_simulation_order(g);
            // order2 = child->get_simulation_order(g);

            STDERR("remove extra syncs...");
            Schedule::remove_redundant_syncs(order1);
            // Schedule::remove_redundant_syncs(order2);
        }

        // distributed order to benchmark to all ranks
        
        if ( 0 == rank ) STDERR("distribute order...");
        mpi_bcast(order1, comm);

#if 0
        {
            std::stringstream ss;
            for (auto &e : order1) {
                ss << e->name() << ", ";
            }
            STDERR(ss.str());
        }
        MPI_Barrier(comm);
#endif

        // benchmark the order
        {
            // warmup
            // BenchOpts opts;
            // Schedule::benchmark(order, comm, opts);
        }
        MPI_Barrier(comm);
        if ( 0 == rank ) STDERR("benchmark...");
        Benchmark::Result benchResult1 =
            benchmarker.benchmark(order1, comm, opts.benchOpts);
        
        MPI_Barrier(comm);
        if (0 == rank) {
            SimResult simres;
            simres.path = order1;
            simres.benchResult = benchResult1;
            result.simResults.push_back(simres);

            STDERR("backprop...");
            child->backprop(ctx, benchResult1);
        }

        if (0 == rank && opts.dumpTreeEvery != 0 && iter % opts.dumpTreeEvery == 0) {
            std::string treePath = "mcts_";
            treePath += std::to_string(iter);
            treePath += ".dot";
            dump_graphviz(treePath, root);
        }

    }

    unregister_handler();
    return result;
}


template <typename Strategy>
Result mcts(const Graph<CpuNode> &g, MPI_Comm comm, const Opts &opts = Opts()) {
    EmpiricalBenchmarker benchmarker;
    return mcts<Strategy>(g, benchmarker, comm, opts);
}

} // namespace mcts