/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include "cast.hpp"
#include "counters.hpp"
#include "graph.hpp"
#include "mcts_node.hpp"
#include "numeric.hpp"
#include "platform.hpp"
#include "schedule.hpp"
#include "trap.hpp"

#include "mpi.h"

#include <chrono>
#include <fstream>
#include <functional>
#include <vector>

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
  Benchmark::Result benchResult;              // times from the simulation
};

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;

struct Result {
  std::vector<SimResult> simResults;
  void dump_csv() const; // dump CSV to stdout
};

/* options for MCTS
 */
struct Opts {
  size_t nIters;              // how many searches to do (0 == infinite)
  bool dumpTree;       // dump the tree dot files every so often
  std::string dumpTreePrefix; // prefix to use for the tree
  bool expandRollout;         // expand the rollout nodes in the tree
  Benchmark::Opts benchOpts;        // options for the runs

  Opts() : dumpTree(true), expandRollout(true) {}
};

template <typename Strategy>
void dump_graphviz(const std::string &path, const Node<Strategy> &root) {

  using Node = Node<Strategy>;

  if (!root.op_) {
    THROW_RUNTIME("attempted to dump empty root");
  }

  STDERR("write " << path);
  std::ofstream os(path);

  bool hideChildrenFullyVisited = true; // don't display children of fully visited
  bool hideChildrenOneRollout = true;   // hide children of nodes with one rollout
  bool hideNoRollouts = true;           // hide nodes with 0 rollouts

  std::function<void(const Node &)> dump_nodes = [&](const Node &node) -> void {
    os << "node_" << &node << " [label=\"";
    os << node.op_->name();
    os << "\n"
       << "rollouts: " << node.n_;
    os << "\n" << node.graphviz_label();
    os << "\"";

    // color fully-visited nodes green
    if (node.fullyVisited_) {
      os << "\nfillcolor=green style=filled";
    } else if (1 == node.n_) {
      os << "\nfillcolor=gray style=filled";
    }

    os << "\n];\n";

    if (hideChildrenFullyVisited && node.fullyVisited_) {
      return;
    }
    if (hideChildrenOneRollout && 1 == node.n_) {
      return;
    }

    for (const auto &child : node.children_) {
      if (child.n_ > 0 || !hideNoRollouts) {
        dump_nodes(child);
      }
    }
  };

  // print the edges from the node to its children
  std::function<void(const Node &)> dump_edges = [&](const Node &node) -> void {
    // if hiding children, the children nodes will not be created so don't draw edges
    if (hideChildrenFullyVisited && node.fullyVisited_) {
      return;
    }
    if (hideChildrenOneRollout && 1 == node.n_) {
      return;
    }

    for (const Node &child : node.children_) {
      if (child.n_ > 0 || !hideNoRollouts) {
        os << "node_" << &node << " -> "
           << "node_" << &child << "\n";
      }
    }

    for (const auto &child : node.children_) {
      if (child.n_ > 0 || !hideNoRollouts) {
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
  enum class Reason { full_tree, large_tree };
  Reason reason_;
  bool value_;
  Stop() : value_(false) {}
  Stop(bool b, const Reason &reason) : reason_(reason), value_(b) {}
  explicit operator bool() const { return value_; }

  const char *c_str() const {
    switch (reason_) {
    case Reason::full_tree:
      return "Full Tree";
    case Reason::large_tree:
      return "Large Tree";
    }
    THROW_RUNTIME("unexpected reason");
  }

  void bcast(int root, MPI_Comm comm) {
    MPI_Bcast(&value_, sizeof(value_), MPI_BYTE, root, comm);
    MPI_Bcast(&reason_, sizeof(reason_), MPI_BYTE, root, comm);
  }
};

template <typename Strategy, typename Benchmarker>
Result mcts(const Graph<OpBase> &g, Platform &plat, Benchmarker &benchmarker,
            const Opts &opts = Opts()) {

  using Context = typename Strategy::Context;
  using Node = Node<Strategy>;

  int rank, size;
  MPI_Comm_rank(plat.comm(), &rank);
  MPI_Comm_size(plat.comm(), &size);

  Node root;
  if (0 == rank) {
    STDERR("create root...");
    root = Node(SCHED_CAST_OR_THROW(BoundOp, g.start_), g);
  }
  MPI_Barrier(plat.comm());

  Result result;

  // print results so far if interrupted
  std::function<void(int)> printResults = [&result](int /*sig*/) -> void { result.dump_csv(); };
  if (0 == rank) {
    register_handler(printResults);
  }

  Context ctx;
  Stop stop;

  // prevent a zillion cudaEventCreate calls
  CudaEventPool eventPool;

  for (size_t iter = 0; 0 == opts.nIters || iter < opts.nIters; ++iter) {

    if (0 == rank) {
      STDERR("iter=" << iter << "/" << opts.nIters << " tree size: " << root.size()
                     << " unvisited size: " << root.unvisited_size()
                     << " fully visisted size: " << root.fully_visited_size());
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
    Sequence<BoundOp> order;

    Node *child = nullptr;    // result of expansion step
    Node *endpoint = nullptr; // result of path expansion
    if (0 == rank) {
      STDERR("select...");
      SCHED_COUNTER_EXPR(double startSelect = MPI_Wtime());
      Node &selected = root.select(ctx);
      SCHED_COUNTER_OP(mcts, SELECT_TIME, += MPI_Wtime() - startSelect);
      STDERR("selected " << selected.op_->desc());

      STDERR("expand...");
      {
        SCHED_COUNTER_EXPR(double start = MPI_Wtime());
        child = &selected.expand(plat);
        SCHED_COUNTER_OP(mcts, EXPAND_TIME, += MPI_Wtime() - start);
      }
      STDERR("expanded to " << child->op_->desc());

      STDERR("rollout...");
      {
        SCHED_COUNTER_EXPR(double start = MPI_Wtime());
        typename Node::RolloutResult rr = child->get_rollout(plat, opts.expandRollout);
        SCHED_COUNTER_OP(mcts, ROLLOUT_TIME, += MPI_Wtime() - start);
        endpoint = rr.backpropStart;
        order = rr.sequence;
      }

      STDERR("remove extra syncs...");
      {
        SCHED_COUNTER_EXPR(double start = MPI_Wtime());
        int n = Schedule::remove_redundant_syncs(order);
        SCHED_COUNTER_OP(mcts, REDUNDANT_SYNC_TIME, += MPI_Wtime() - start);
        STDERR("removed " << n << " sync operations");
      }
    }

    // distributed order to benchmark to all ranks
    if (0 == rank)
      STDERR("bcast sequence");
    order = mpi_bcast(order, g, plat.comm());

    // provision resources for this program
    {
      SCHED_COUNTER_EXPR(double start = MPI_Wtime());
      eventPool.reset();
      ResourceMap rMap;
      {
        std::vector<HasEvent *> ops;

        for (const auto &op : order) {
          if (HasEvent *he = dynamic_cast<HasEvent *>(op.get())) {
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
      SCHED_COUNTER_OP(mcts, RMAP_TIME, += MPI_Wtime() - start);
    }

    MPI_Barrier(plat.comm());
    if (0 == rank)
      STDERR("benchmark...");
    Benchmark::Result br1;
    {
      SCHED_COUNTER_EXPR(double start = MPI_Wtime());
      br1 = benchmarker.benchmark(order, plat, opts.benchOpts);
      MPI_Barrier(plat.comm());
      SCHED_COUNTER_OP(mcts, BENCHMARK_TIME, += MPI_Wtime() - start);
    }
    if (0 == rank) {
      STDERR("01=" << br1.pct01 << " 10=" << br1.pct10);
    }

    MPI_Barrier(plat.comm());
    if (0 == rank) {
      SimResult simres;
      simres.path = order;
      simres.benchResult = br1;
      result.simResults.push_back(simres);

      STDERR("backprop...");
      {
        SCHED_COUNTER_EXPR(double start = MPI_Wtime());
        endpoint->backprop(ctx, br1);
        SCHED_COUNTER_OP(mcts, BACKPROP_TIME, += MPI_Wtime() - start);
      }
    }

    // FIXME: segfault here?
    if (0 == rank && (opts.dumpTree && 
                      (iter < 10 || iter >= 10 && iter < 50 && iter % 10 == 0 ||
                       iter >= 50 && iter < 100 && iter % 25 == 0))) {
      std::string treePath = opts.dumpTreePrefix;
      treePath += "mcts_";
      treePath += std::to_string(iter);
      treePath += ".dot";
      dump_graphviz(treePath, root);
    }

    if (0 == rank) {
      SCHED_COUNTER_EXPR(STDERR("mcts.SELECT_TIME " << counters::mcts.SELECT_TIME));
      SCHED_COUNTER_EXPR(STDERR("mcts.EXPAND_TIME " << counters::mcts.EXPAND_TIME));
      SCHED_COUNTER_EXPR(STDERR("mcts.ROLLOUT_TIME " << counters::mcts.ROLLOUT_TIME));
      SCHED_COUNTER_EXPR(STDERR("mcts.REDUNDANT_SYNC_TIME " << counters::mcts.REDUNDANT_SYNC_TIME));
      SCHED_COUNTER_EXPR(STDERR("mcts.RMAP_TIME " << counters::mcts.RMAP_TIME));
      SCHED_COUNTER_EXPR(STDERR("mcts.BENCHMARK_TIME " << counters::mcts.BENCHMARK_TIME));
      SCHED_COUNTER_EXPR(STDERR("mcts.BACKPROP_TIME " << counters::mcts.BACKPROP_TIME));
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