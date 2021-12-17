#pragma once

#include "graph.hpp"
#include "schedule.hpp"
#include "trap.hpp"
#include "numeric.hpp"
#include "mcts_node.hpp"
#include "platform.hpp"
#include "cast.hpp"

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
    std::vector<std::shared_ptr<BoundOp>> path; // path that is simulated
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

/* broadcast `order` from rank 0 to the other ranks
*/
std::vector<std::shared_ptr<BoundOp>> mpi_bcast(
    const std::vector<std::shared_ptr<BoundOp>> &order,
    const Graph<OpBase> &g,
    MPI_Comm comm
);

template<typename Strategy>
void dump_graphviz(const std::string &path, const Node<Strategy> &root) {

    using Node = Node<Strategy>;

    STDERR("write " << path);
    std::ofstream os(path);

    std::function<void(const Node &)> dump_nodes = [&](const Node &node) -> void {
        os << "node_" << &node << " [label=\"";
        os << node.op_->name();
        os << "\n" << node.n_;
        os << "\n" << node.state_.graphviz_label_line();
        os << "\"];\n";

        for (const auto &child : node.children_) {
            if (child.n_ > 0) {
                dump_nodes(child);
            }
        }
    };

    std::function<void(const Node &)> dump_edges = [&](const Node &node) -> void {
        for (const Node &child : node.children_) {
            if (child.n_ > 0) {
                os << "node_" << &node << " -> " << "node_" << &child << "\n";
            }
        }
        for (const auto &child : node.children_) {
            if (child.n_ > 0) {
                dump_edges(child);
            }
        }
    };

    os << "digraph D {";
    dump_nodes(root);
    dump_edges(root);
    os << "}\n";
}

/* a stop signal to share between ranks */
struct Stop {
    enum class Reason {
        full_tree,
        large_tree
    };
    Reason reason_;
    bool value_;
    Stop() : value_(false) {}
    Stop(bool b, const Reason &reason) : reason_(reason), value_(b) {}
    explicit operator bool() const {return value_; }

    const char * c_str() const {
        switch(reason_) {
            case Reason::full_tree: return "Full Tree";
            case Reason::large_tree: return "Large Tree";
        }
        THROW_RUNTIME("unexpected reason");
    }

    void bcast(int root, MPI_Comm comm) {
        MPI_Bcast(&value_, sizeof(value_), MPI_BYTE, root, comm);
        MPI_Bcast(&reason_, sizeof(reason_), MPI_BYTE, root, comm);
    }
};


template <typename Strategy, typename Benchmarker>
Result mcts(
    const Graph<OpBase> &g, 
    Platform &plat,
    Benchmarker &benchmarker, 
    const Opts &opts = Opts()) {

    using Context = typename Strategy::Context;
    using Node = Node<Strategy>;

    int rank, size;
    MPI_Comm_rank(plat.comm(), &rank);
    MPI_Comm_size(plat.comm(), &size);

    
    Node root(nullptr);
    if (0 == rank) {
        STDERR("create root...");
        root = SCHED_CAST_OR_THROW(BoundOp, g.start_);
    }
    MPI_Barrier(plat.comm());

    Result result;

    // print results so far if interrupted
    std::function<void(int)> printResults = [&result](int sig) -> void {
        (void) sig;
        for (const auto &simres : result.simResults) {
            std::cout << simres.benchResult.pct10;
            for (const auto &op : simres.path) {
                std::cout << "|" << op->json();
            }
            std::cout << "\n"; 
        }
    };
    if (0 == rank) {
        register_handler(printResults);
    }

    Context ctx;
    Stop stop;   

    // prevent a zillion cudaEventCreate calls
    CudaEventPool eventPool;

    for (size_t iter = 0; iter < opts.nIters; ++iter) {

        if (0 == rank) {
            STDERR("iter=" << iter << "/" << opts.nIters << " tree size: " << root.size());
        }

        if (root.fullyVisited_) {
            stop = Stop(true, Stop::Reason::full_tree);
        }
        stop.bcast(0, plat.comm());
        if (bool(stop)) {
            STDERR("Stop requested: " << stop.c_str());
            break;
        }

        // the order the nodes will be executed
        std::vector<std::shared_ptr<BoundOp>> order;


        Node *child = nullptr; // result of expansion step
        Node *endpoint = nullptr; // result of path expansion
        if (0 == rank) {
            STDERR("select...");
            Node &selected = root.select(ctx, g);
            STDERR("selected " << selected.op_->name());

            STDERR("expand...");
            child = &selected.expand(ctx, plat, g);
            STDERR("expanded to " << child->op_->name());

            STDERR("simulate...");
            order = child->get_simulation_order(plat, g);

            STDERR("remove extra syncs...");
            {
                int n = Schedule::remove_redundant_syncs(order);
                STDERR("removed " << n << " sync operations");
            }

            STDERR("expand simulation path");
            endpoint = &root.expand_order(ctx, plat, g, order);
        }

        // distributed order to benchmark to all ranks
        if ( 0 == rank ) STDERR("distribute order...");
        order = mpi_bcast(order, g, plat.comm());

        // assemble a resource map for this program
        eventPool.reset();
        ResourceMap rMap;
        {
            std::vector<HasEvent*> ops;

            for (const auto &op : order) {
                if (HasEvent *he = dynamic_cast<HasEvent*>(op.get())) {
                    ops.push_back(he);
                }
            }

            for (HasEvent *op : ops) {
                for (Event event : op->get_events()) {
                    if (!rMap.contains(event)) {
                        rMap.insert(event, eventPool.new_event());
                    }
                }
            }
        }
        plat.resource_map() = rMap;

        // benchmark the order
        {
            // warmup
            // BenchOpts opts;
            // Schedule::benchmark(order, plat.comm(), opts);
        }
        MPI_Barrier(plat.comm());
        if ( 0 == rank ) STDERR("benchmark...");
        Benchmark::Result br1 =
            benchmarker.benchmark(order, plat, opts.benchOpts);
        MPI_Barrier(plat.comm());
        if ( 0 == rank ) {
            STDERR("01=" << br1.pct01 << " 10=" << br1.pct10);
        }
        
        MPI_Barrier(plat.comm());
        if (0 == rank) {
            SimResult simres;
            simres.path = order;
            simres.benchResult = br1;
            result.simResults.push_back(simres);

            STDERR("backprop...");
            endpoint->backprop(ctx, br1);
        }

        if (0 == rank && opts.dumpTreeEvery != 0 && iter % opts.dumpTreeEvery == 0) {
            std::string treePath = "mcts_";
            treePath += std::to_string(iter);
            treePath += ".dot";
            dump_graphviz(treePath, root);
        }

    }
    MPI_Barrier(plat.comm());

    unregister_handler();
    return result;
}


template <typename Strategy>
Result mcts(const Graph<OpBase> &g, MPI_Comm comm, const Opts &opts = Opts()) {
    EmpiricalBenchmarker benchmarker;
    return mcts<Strategy>(g, benchmarker, comm, opts);
}





} // namespace mcts