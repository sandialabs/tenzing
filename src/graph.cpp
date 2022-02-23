/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "sched/graph.hpp"
#include "sched/macro_at.hpp"
#include "sched/cuda/ops_cuda.hpp"

#include <mpi.h>

#include <fstream>
#include <sstream>

template <> void Graph<OpBase>::dump_graphviz(const std::string &path) const {

  STDERR("write " << path);
  std::ofstream os(path);
  os << "digraph D {";

  // dump nodes
  for (const auto &kv : succs_) {
    os << "op_" << kv.first.get() << " [label=\"";
    os << kv.first->name();

    if (auto ss = std::dynamic_pointer_cast<BoundGpuOp>(kv.first)) {
      os << "\nstream " << ss->stream();
    }

    os << "\"];\n";
  }

  // dump edges
  for (const auto &kv : succs_) {
    for (const auto &succ : kv.second) {
      os << "op_" << kv.first.get() << " -> "
         << "op_" << succ.get() << "\n";
    }
  }

  os << "}\n";
}

std::vector<Graph<OpBase>> use_streams(const Graph<OpBase> &orig,
                                       const std::vector<Stream::id_t> &streams) {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  using op_t = std::shared_ptr<OpBase>;
  using gpu_t = std::shared_ptr<GpuOp>;

  std::vector<Graph<OpBase>> graphlist;
  std::vector<Graph<OpBase>> ret;

  graphlist.push_back(orig);

  while (!graphlist.empty()) {
    if (0 == rank) {
      std::cerr << "graphlist.size() = " << graphlist.size() << "\n";
    }

    // work from the back of the list.
    Graph<OpBase> g = graphlist.back();
    graphlist.pop_back();

    // find a GpuNode in the graph
    bool hasGpuNode = false;
    for (auto &kv : g.succs_) {
      op_t n = kv.first;
      if (gpu_t gpu = std::dynamic_pointer_cast<GpuOp>(n)) {

        // create a copy of that graph, with the GPU node replaced by a StreamedNode for each stream
        for (Stream::id_t stream : streams) {

          // get a copy of the gpu node. we know it's a GPU node so cast away
          auto copy = std::shared_ptr<GpuOp>(static_cast<GpuOp *>(gpu->clone().release()));
          if (!copy)
            throw std::runtime_error(AT);

          auto streamed = std::make_shared<BoundGpuOp>(copy, stream);
          Graph<OpBase> ng = g.clone_but_replace(streamed, gpu);
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

/* outputs a clone of orig, except gpuOp[i] is assigned to streams[assignments[i]]
 */
Graph<OpBase> apply_assignment(const Graph<OpBase> &orig,
                               const std::vector<std::shared_ptr<GpuOp>> &gpuOps,
                               const std::vector<Stream::id_t> &streams,
                               const std::vector<int> assignments) {
  using gpu_t = std::shared_ptr<GpuOp>;

  if (assignments.size() != gpuOps.size()) {
    THROW_RUNTIME("expected one assignment per gpuOp");
  }

  Graph<OpBase> ng(orig);

  // replace each GPU node with a streamedNode
  for (size_t ai = 0; ai < assignments.size(); ++ai) {
    gpu_t gpu = gpuOps[ai];
    auto copy = std::shared_ptr<GpuOp>(static_cast<GpuOp *>(gpu->clone().release()));
    if (!copy)
      THROW_RUNTIME("should have been a gpu node");

    size_t si = assignments[ai];
    if (si >= streams.size())
      THROW_RUNTIME("stream index too large");
    Stream::id_t stream = streams[si];
    auto streamed = std::make_shared<BoundGpuOp>(copy, stream);
    ng.replace(gpu, streamed);
  }
  return ng;
}

/*
The assignment strategy follows

The dependence of operations does not matter, it's just the combinatorial assignment of operations
to resources. For example, two resource types: Streams S0,S1 and CPUs C0, C1, C2

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
std::vector<Graph<OpBase>> use_streams2(const Graph<OpBase> &orig,
                                        const std::vector<Stream::id_t> &streams) {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  using op_t = std::shared_ptr<OpBase>;
  using gpu_t = std::shared_ptr<GpuOp>;

  // extract all GPU operations
  std::vector<gpu_t> gpuOps;

  for (auto &kv : orig.succs_) {
    op_t n = kv.first;
    if (gpu_t gpu = std::dynamic_pointer_cast<GpuOp>(n)) {
      gpuOps.push_back(gpu);
    }
  }

  // each assignment is a vector of which resource each gpuOp is assigned to
  std::vector<std::vector<int>> assignments;

  size_t numAssignments = 1;
  for (size_t i = 0; i < gpuOps.size(); ++i) {
    numAssignments *= std::min(i + 1, streams.size());
  }

  std::cerr << "creating " << numAssignments << " assignments for " << gpuOps.size()
            << " operations in " << streams.size() << " streams\n";

  for (size_t ai = 0; ai < numAssignments; ++ai) {

    std::vector<int> assignment;
    int div = numAssignments;
    for (size_t gi = 0; gi < gpuOps.size(); ++gi) {
      int numOptions = std::min(gi + 1, streams.size());
      div /= numOptions;

      // std::cerr << "ass " << ai << ": idx=" << gi << " div=" << div << " mod=" << numOptions <<
      // "\n";

      assignment.push_back((ai / div) % numOptions);
    }
    assignments.push_back(assignment);
  }

  std::vector<Graph<OpBase>> ret;

  for (const auto &assignment : assignments) {

#if 0
        // for (auto &e : assignment) {
        //     std::cerr << " " << e;
        // }
        // std::cerr << "\n";

        // get a copy of the graph with all the same nodes
        Graph<OpBase> ng(orig);

        // replace each GPU node with a streamedNode
        for (size_t ai = 0; ai < assignment.size(); ++ai) {
            gpu_t gpu = gpuOps[ai];
            auto copy = std::shared_ptr<GpuOp>(static_cast<GpuNode*>(gpu->clone().release()));
            if (!copy) THROW_RUNTIME("should have been a gpu node");

            size_t si = assignment[ai];
            if (si >= streams.size()) THROW_RUNTIME("stream index too large");
            Stream::id_t stream = streams[si];
            auto streamed = std::make_shared<BoundGpuOp>(copy, stream);
            ng.replace(gpu, streamed);
        }
        ret.push_back(ng);
#else
    ret.push_back(apply_assignment(orig, gpuOps, streams, assignment));
#endif
  }

  return ret;
}

bool is_equivalent_stream_mapping(const Graph<OpBase> &a, const Graph<OpBase> &b) {

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::map<Stream, Stream> bij; // stream b for stream a

  // if a's stream matches b's under bijection, or new bijection entry,
  // return true. else return false.
  auto check_or_update_bijection = [&](const std::shared_ptr<OpBase> &_a,
                                       const std::shared_ptr<OpBase> &_b) -> bool {
    auto aa = std::dynamic_pointer_cast<BoundGpuOp>(_a);
    auto bb = std::dynamic_pointer_cast<BoundGpuOp>(_b);
    if (aa && bb) {
      if (bij.count(aa->stream()) && bb->stream() != bij[aa->stream()]) {
        return false;
      }
      if (bij.count(bb->stream()) && aa->stream() != bij[bb->stream()]) {
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
  for (; ai != a.succs_.end() && bi != b.succs_.end(); ++ai, ++bi) {

    const auto u_a = ai->first;
    const auto u_b = bi->first;

#if 0
        if (0 == rank) std::cerr << "compare " << u_a->name() << " vs. " << u_b->name() << "\n";
#endif
    if (!u_a->eq(u_b)) { // not same operation
      // if (0 == rank) std::cerr << "FALSE: unequal operations: " << u_a->name() << " vs. " <<
      // u_b->name() << "\n";
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
