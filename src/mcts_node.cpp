/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "sched/mcts_node.hpp"

#include "sched/operation.hpp"
#include "sched/cuda/ops_cuda.hpp"

// true iff unbound version of e in unbound versions of v
bool unbound_contains(const std::vector<std::shared_ptr<BoundOp>> &v,
                      const std::shared_ptr<OpBase> &e) {

  // unbound version if bound
  std::shared_ptr<OpBase> ue;
  if (auto bgo = std::dynamic_pointer_cast<BoundGpuOp>(e)) {
    ue = bgo->unbound();
  } else {
    ue = e;
  }

  for (const auto &ve : v) {

    // get unbound version if bound
    std::shared_ptr<OpBase> uve;
    if (auto bgo = std::dynamic_pointer_cast<BoundGpuOp>(ve)) {
      uve = bgo->unbound();
    } else {
      uve = ve;
    }

    if (uve->eq(ue)) {
      return true;
    }
  }
  return false;
}

// true iff unbound version of e in unbound versions of v
Sequence<BoundOp>::const_iterator
unbound_find(const Sequence<BoundOp> &v, const std::shared_ptr<OpBase> &e) {

  // unbound version if bound
  std::shared_ptr<OpBase> ue;
  if (auto bgo = std::dynamic_pointer_cast<BoundGpuOp>(e)) {
    ue = bgo->unbound();
  } else {
    ue = e;
  }

  for (auto it = v.begin(); it < v.end(); ++it) {

    // get unbound version if bound
    std::shared_ptr<OpBase> uve;
    if (auto bgo = std::dynamic_pointer_cast<BoundGpuOp>(*it)) {
      uve = bgo->unbound();
    } else {
      uve = *it;
    }

    if (uve->eq(ue)) {
      return it;
    }
  }
  return v.end();
}

// uses cudaEventRecord, cudaStreamWait and cudaEventSync for synchronization
struct EventSynchronizer {

  // find a in v or return v.end()
  static Sequence<BoundOp>::const_iterator
  find(const Sequence<BoundOp> &v, const std::shared_ptr<OpBase> &a) {
    for (auto it = v.begin(); it != v.end(); ++it) {
      if (a->eq(*it)) {
        return it;
      }
    }
    return v.end();
  }

  /* for a -> cudaEventRecord -> cudaEventSync -> b,
     true if path contains appropriate a ... CER ... CES to sync with b
  */
  static bool is_synced_gpu_then_cpu(const std::shared_ptr<BoundGpuOp> &a,
                                     const std::shared_ptr<CpuOp> & /*b*/,
                                     const Sequence<BoundOp> &path) {
    // find a
    auto ai = find(path, a);
    if (path.end() == ai) {
      THROW_RUNTIME("couldn't find a " << a->name() << " in path");
    }

    // check for any existing CER ... CES combo
    for (auto ceri = ai; ceri < path.end(); ++ceri) {
      if (auto cer = std::dynamic_pointer_cast<CudaEventRecord>(*ceri)) {
        if (cer->stream() == a->stream()) {
          for (auto cesi = ceri + 1; cesi < path.end(); ++cesi) {
            if (auto ces = std::dynamic_pointer_cast<CudaEventSync>(*cesi)) {
              if (ces->event() == cer->event()) {
                return true;
              }
            }
          }
        }
      }
    }
    return false;
  }

  // for a -> cudaEventRecord -> cudaStreamWaitEvent -> b
  // true if path contains appropriate a ... CER ... CSWE to sync with b
  static bool is_synced_gpu_then_gpu(const std::shared_ptr<BoundGpuOp> &a,
                                     const std::shared_ptr<BoundGpuOp> &b,
                                     const Sequence<BoundOp> &path) {

    STDERR("is_synced_gpu_then_gpu for " << a->desc() << " -> " << b->desc());

    // implicitly synced already if in the same stream
    if (a->stream() == b->stream()) {
      return true;
    }

    // find a
    auto ai = find(path, a);
    if (path.end() == ai) {
      THROW_RUNTIME("couldn't find a " << a->name() << " in path");
    }

    // check all CERs that sync with a
    for (auto it = ai; it != path.end(); ++it) {
      if (auto cer = std::dynamic_pointer_cast<CudaEventRecord>(*it)) {
        if (cer->stream() == a->stream()) {
          STDERR(cer->desc() << " records a: " << a->desc());

          // synced if there is an approprate CSWE
          for (auto wi = it; wi < path.end(); ++wi) {
            if (auto cswe = std::dynamic_pointer_cast<CudaStreamWaitEvent>(*wi)) {
              if (cswe->event() == cer->event() && cswe->stream() == b->stream()) {
                STDERR(cer->desc() << " makes b: " << b->desc() << " wait for a: " << a->desc());
                return true;
              }
            }
          }
        }
      }
    }
    return false;
  }

  // return the next sync missing in the chain from a -> cudaEventRecord -> cudaEventSync -> b
  // there may be multiple cuda event records for a in the path.
  // check all of them to see if there is a CES
  // if not, emit a sync for the first CER
  // return is falsy if no sync is needed
  // FIXME: should emit a sync for all records, and clean up later?
  static std::shared_ptr<BoundOp>
  make_sync_gpu_then_cpu(const std::shared_ptr<BoundGpuOp> &a, const std::shared_ptr<CpuOp> &b,
                         const Sequence<BoundOp> &path) {
    // find the GPU operation on the path
    auto ai = find(path, a);
    if (path.end() == ai) {
      THROW_RUNTIME("counldn't find a " << a->name() << " in path");
    }

    // find the first CER for a
    std::shared_ptr<CudaEventRecord> firstCER;
    for (auto it = ai; it != path.end(); ++it) {
      if (auto cer = std::dynamic_pointer_cast<CudaEventRecord>(*it)) {
        if (cer->stream() == a->stream()) {
          firstCER = cer;
          break;
        }
      }
    }

    // nothing to do if there is already a sync
    for (auto it = ai; it != path.end(); ++it) {
      if (auto cer = std::dynamic_pointer_cast<CudaEventRecord>(*it)) {
        if (cer->stream() == a->stream()) {
          // check for existing CES
          for (auto cssi = it + 1; cssi < path.end(); ++cssi) {
            if (auto css = std::dynamic_pointer_cast<CudaEventSync>(*cssi)) {
              if (css->event() == cer->event()) {
                return std::shared_ptr<BoundOp>(); // falsy
              }
            }
          }
        }
      }
    }

    if (firstCER) {
      auto CES = std::make_shared<CudaEventSync>(firstCER->event());
      CES->update_name({}, {b});
      return CES;
    } else {
      Event event = path.new_unique_event();
      auto CER = std::make_shared<CudaEventRecord>(event, a->stream());
      CER->update_name({a}, {});
      return CER;
    }
  }

  // return the next sync missing in the chain from a -> cudaEventRecord -> cudaStreamWaitEvent -> b
  // there may be multiple CER for a in the path.
  // check all of them to see if there is a CSWE
  // if not, emit a CSWE for the first CER to waiting for unrelated ops
  // return is falsy if no sync is needed
  static std::shared_ptr<BoundOp>
  make_sync_gpu_then_gpu(const std::shared_ptr<BoundGpuOp> &a, const std::shared_ptr<BoundGpuOp> &b,
                         const Sequence<BoundOp> &path) {

    if (a->stream() == b->stream()) {
      return std::shared_ptr<BoundOp>();
    }

    // find a
    auto ai = find(path, a);
    if (path.end() == ai) {
      THROW_RUNTIME("couldn't find a " << a->name() << " in path");
    }

    // look for first cer following a
    std::shared_ptr<CudaEventRecord> firstCER;
    for (auto it = ai; it != path.end(); ++it) {
      if (auto p = std::dynamic_pointer_cast<CudaEventRecord>(*it)) {
        if (p->stream() == a->stream()) {
          firstCER = p;
          break;
        }
      }
    }

    // nothing to do if there is already an appropriate CER ... CSWE pair
    for (auto ceri = ai; ceri != path.end(); ++ceri) {
      if (auto cer = std::dynamic_pointer_cast<CudaEventRecord>(*ceri)) {
        if (cer->stream() == a->stream()) {
          // check for existing CES
          for (auto cswei = ceri + 1; cswei < path.end(); ++cswei) {
            if (auto cswe = std::dynamic_pointer_cast<CudaStreamWaitEvent>(*cswei)) {
              if (cswe->event() == cer->event() && cswe->stream() == b->stream()) {
                return std::shared_ptr<BoundOp>();
              }
            }
          }
        }
      }
    }

    // if no existing CER ... CSWE, then emit one for the first CER, or emit a new CER
    if (firstCER) {
      // assume there is no CudaStreamWaitEvent, so produce one that synces b's stream with
      auto CSWE = std::make_shared<CudaStreamWaitEvent>(b->stream(), firstCER->event());
      CSWE->update_name({a}, {});
      return CSWE;
    } else {
      Event event = path.new_unique_event();
      auto CER = std::make_shared<CudaEventRecord>(event, a->stream());
      CER->update_name({}, {b});
      return CER;
    }
  }

public:
  // true iff bo is in path and is synced with all preds in path
  static bool is_synced(const std::shared_ptr<BoundOp> &bo, const Graph<OpBase> &g,
                        const Sequence<BoundOp> &path) {

    // graph may contain bo or the unbound version of bo
    auto it = g.preds_find_or_find_unbound(bo);
    if (it == g.preds_.end()) {
      THROW_RUNTIME("couldn't find BoundOp " << bo->name() << " in graph");
    }

    // find all ops on path that are predecessors of bo
    for (const auto &gPred : it->second) { // predecessor in the graph

      // find the predecessor in the path
      auto predi = unbound_find(path, gPred);
      if (path.end() == predi) {
        THROW_RUNTIME("couldn't find " << gPred->desc() << " in path (pred of " << bo->desc()
                                       << ")");
      }
      const std::shared_ptr<BoundOp> pred = *predi;

      STDERR("is_synced: is " << bo->desc() << " synced with pred " << pred->desc() << "?");

      // various CPU/GPU sync combinations
      // predicates are check in the graph, so they're not Bound
      auto bCpu = std::dynamic_pointer_cast<CpuOp>(bo);
      auto bGpu = std::dynamic_pointer_cast<BoundGpuOp>(bo);
      auto pCpu = std::dynamic_pointer_cast<CpuOp>(pred);
      auto pGpu = std::dynamic_pointer_cast<BoundGpuOp>(pred);
      bool pS = bool(std::dynamic_pointer_cast<Start>(pred));

      if (pS) {                  // pred is start node, no need to sync
        ;                        // no need to sync with this pred
      } else if (pCpu && bCpu) { // cpu -> cpu (nothing)
        ;                        // no sync needed
      } else if (pGpu && bCpu) { // gpu -> cpu (CER & CEW)
        // auto pBound = std::dynamic_pointer_cast<BoundGpuOp>(pred);
        // if (!pBound) THROW_RUNTIME("couldn't get BoundGpuOp for " << pred->desc());
        if (!is_synced_gpu_then_cpu(pGpu, bCpu, path)) {
          return false;
        }
      } else if (pCpu && bGpu) { // cpu -> gpu
        ;                        // no sync needed
      } else if (pGpu && bGpu) { // gpu -> gpu (maybe CER & CSW)
        // auto pBound = std::dynamic_pointer_cast<BoundGpuOp>(bo);
        // auto bBound = std::dynamic_pointer_cast<BoundGpuOp>(pred);
        if (!is_synced_gpu_then_gpu(pGpu, bGpu, path)) {
          return false;
        }
      } else {
        std::stringstream ss;
        ss << "pc=" << bool(pCpu);
        ss << " pg=" << bool(pGpu);
        ss << " bc=" << bool(bCpu);
        ss << " bg=" << bool(bGpu);
        THROW_RUNTIME("unexpected op combination: " << pred->name() << " and " << bo->name() << ": "
                                                    << ss.str());
      }
    }
    return true;
  }

  // return any operations to insert after `path` that would help synchronize `bo` with its
  // predecessors may return empty vector, in which case bo is synchronized with preds
  static std::vector<std::shared_ptr<BoundOp>>
  make_syncs(const std::shared_ptr<BoundOp> &bo, const Graph<OpBase> &g,
             const Sequence<BoundOp> &path, bool quiet = true) {

    // graph may contain bo or the unbound version of bo
    auto it = g.preds_find_or_find_unbound(bo);
    if (it == g.preds_.end()) {
      THROW_RUNTIME("couldn't find BoundOp " << bo->name() << " in graph");
    }

    STDERR("make syncs for " << bo->desc());
    std::vector<std::shared_ptr<BoundOp>> syncs;

    // find all ops on path that are predecessors of bo
    for (const auto &gPred : it->second) {

      if (!quiet)
        STDERR("graph pred " << gPred->desc() << " of " << bo->desc() << "...");

      // find the predecessor in the path
      auto predi = unbound_find(path, gPred);
      if (path.end() == predi) {
        THROW_RUNTIME("couldn't find " << gPred->desc() << " in path");
      }
      const std::shared_ptr<BoundOp> pred = *predi;
      if (!quiet)
        STDERR("pred " << pred->desc() << " of " << bo->desc() << "...");

      // various CPU/GPU sync combinations
      auto bCpu = std::dynamic_pointer_cast<CpuOp>(bo);
      auto bGpu = std::dynamic_pointer_cast<BoundGpuOp>(bo);
      auto pCpu = std::dynamic_pointer_cast<CpuOp>(pred);
      auto pGpu = std::dynamic_pointer_cast<BoundGpuOp>(pred);
      bool pS = bool(std::dynamic_pointer_cast<Start>(pred));

      if (pS) {                  // pred is start node
        ;                        // no sync
      } else if (pCpu && bCpu) { // cpu -> cpu (nothing)
        ;                        // no sync needed
      } else if (pGpu && bCpu) { // gpu -> cpu (CER & CEW)
                                 // auto pBound = std::dynamic_pointer_cast<BoundGpuOp>(pred);
                                 // if (!is_synced_gpu_then_cpu(pGpu, bCpu, path)) {
        auto syncer = make_sync_gpu_then_cpu(pGpu, bCpu, path);
        if (syncer) {
          STDERR("adding " << syncer->desc() << " to sync " << bCpu->desc() << " after "
                           << pGpu->desc());
          syncs.push_back(syncer);
        }
        // }
      } else if (pCpu && bGpu) { // cpu -> gpu
        ;                        // no sync needed
      } else if (pGpu && bGpu) { // gpu -> gpu (maybe CER & CSW)
                                 // auto pBound = std::dynamic_pointer_cast<BoundGpuOp>(bo);
                                 // auto bBound = std::dynamic_pointer_cast<BoundGpuOp>(pred);
                                 // if (!is_synced_gpu_then_gpu(pGpu, bGpu, path)) {
        auto syncer = make_sync_gpu_then_gpu(pGpu, bGpu, path);
        if (syncer) {
          STDERR("adding " << syncer->desc() << " to sync " << bGpu->desc() << " after "
                           << pGpu->desc());
          syncs.push_back(syncer);
        }
        // }
      } else {
        THROW_RUNTIME("unpected Op combination");
      }
    }

    // FIXME: there may be duplicate syncs here, since the sync detection has
    // some logic to choose between two different event records that follow the same event
    // e.g., two ops in the same stream have two different event records,
    // and a succ of those ops will pick up the later event record for both ops

    for (auto si = syncs.begin(); si < syncs.end(); ++si) {
      for (auto sj = si + 1; sj < syncs.end(); ++sj) {
        if ((*si)->eq(*sj)) {
          // sj should be after si, but it is about to be incremented in the loop
          si = sj = syncs.erase(si);
          STDERR("erased a redundant generated sync");
        }
      }
    }

    return syncs;
  }
};

/* return the frontier of nodes from g given already-traversed nodes
    g may or may not include:
      synchronization
      resource assignments

    The next possible operations are those that have:
      all predecessor issued
      are not in the completed operations

    For those operations, all possible resource assignments can be considered
    The next executable states are those BoundOps that have
        the resources of all predecessors synced

    If a predecessor is issued but its resources are not synced,
    the fronter can contain the corresponding sync operation instead of the operation itself

    The next executable states are those operations that have
      all predecessors issued
      all resources for those predecessors

    FIXME: this adds synchronizations for all possible stream assignments of future
    operations, of which only one is selected, usually inserting lots of extra syncs

*/
std::vector<std::shared_ptr<BoundOp>>
get_frontier(Platform &plat, const Graph<OpBase> &g,
             const Sequence<BoundOp> &completed) {
  typedef EventSynchronizer Synchronizer;
  /*
  find candidate operations for the frontier
      all predecessors in `completed`
      is not itself in `completed`
  */
  {
    std::stringstream ss;
    ss << "frontier for state: ";
    for (const auto &op : completed) {
      ss << op->desc() << ",";
    }
    STDERR(ss.str());
  }

  STDERR("consider ops with >= 1 pred completed...");
  std::vector<std::shared_ptr<OpBase>> onePredCompleted;
  for (const auto &cOp : completed) {
    STDERR("...succs of " << cOp->desc() << " (@" << cOp.get() << ")");

    // some nodes in the path will not be in the graph (inserted syncs)
    // other nodes in the path are bound versions of that in the graph

    auto it = g.succs_find_or_find_unbound(cOp);
    if (g.succs_.end() != it) {

      // all successors of a completed op have at least one pred completed
      for (const auto &succ : it->second) {
        // don't add duplicates
        if (onePredCompleted.end() ==
            std::find(onePredCompleted.begin(), onePredCompleted.end(), succ)) {
          onePredCompleted.push_back(succ);
        }
      }
    }
  }

  {
    std::stringstream ss;
    ss << "one pred completed: ";
    for (const auto &op : onePredCompleted) {
      ss << op->desc() << ",";
    }
    STDERR(ss.str());
  }

  STDERR("reject ops already done or with incomplete preds...");
  std::vector<std::shared_ptr<OpBase>> candidates;
  for (const auto &cOp : onePredCompleted) {
    // reject ops that we've already done
    if (completed.contains_unbound(cOp)) {
      STDERR(cOp->name() << " already done");
      continue;
    }

    // reject ops that all preds are not done
    bool allPredsCompleted = true;
    for (const auto &pred : g.preds_.at(cOp)) {
      if (!completed.contains_unbound(pred)) {
        STDERR(cOp->name() << " missing pred " << pred->name());
        allPredsCompleted = false;
        break;
      }
    }
    if (!allPredsCompleted) {
      STDERR(cOp->name() << " missing a pred");
      continue;
    }
    candidates.push_back(cOp);
  }

  {
    std::stringstream ss;
    ss << "preds complete AND not done: ";
    for (const auto &op : candidates) {
      ss << op->desc() << ",";
    }
    STDERR(ss.str());
  }

  std::vector<std::shared_ptr<BoundOp>> frontier;

  STDERR("generate frontier from candidates...");
  // candidates may or may not be assigned to resources
  // get the viable resource assignments for the candidates
  for (const auto &candidate : candidates) {
    STDERR("candidate " << candidate->desc() << "...");
    std::vector<std::shared_ptr<BoundOp>> bounds = make_platform_variations(plat, candidate);
    STDERR("...got " << bounds.size() << " platform variations");

    for (const std::shared_ptr<BoundOp> &bound : bounds) {
      // if the candidate is already synchronized with its preds, it can be added to the frontier
      if (Synchronizer::is_synced(bound, g, completed)) {
        STDERR("variation of " << bound->desc() << " is synced");
        frontier.push_back(bound);
      } else { // otherwise synchronizers should be added instead
        STDERR("variation of " << bound->desc() << " is not synced with preds");
        std::vector<std::shared_ptr<BoundOp>> syncs = Synchronizer::make_syncs(bound, g, completed);
        STDERR("adding synchronizers for " << bound->desc() << " to frontier:");
        for (const auto &sync : syncs) {
          STDERR(sync->desc());
          frontier.push_back(sync);
        }
      }
    }
  }

  keep_uniques(frontier);
  return frontier;
}

std::vector<std::shared_ptr<BoundOp>>
mcts::get_graph_frontier(Platform &plat, const Graph<OpBase> &g,
                         const Sequence<BoundOp> &completed, bool quiet) {

  /*
  find candidate operations for the frontier
      all predecessors in `completed`
      is not itself in `completed`
  */
  {
    std::stringstream ss;
    ss << "frontier for state: ";
    for (const auto &op : completed) {
      ss << op->desc() << ",";
    }
    STDERR(ss.str());
  }

  STDERR("consider ops with >= 1 pred completed...");
  std::vector<std::shared_ptr<OpBase>> onePredCompleted;
  for (const auto &cOp : completed) {
    if (!quiet)
      STDERR("...succs of " << cOp->desc() << " (@" << cOp.get() << ")");

    // some nodes in the path will not be in the graph (inserted syncs)
    // other nodes in the path are bound versions of that in the graph

    auto it = g.succs_find_or_find_unbound(cOp);
    if (g.succs_.end() != it) {

      // all successors of a completed op have at least one pred completed
      for (const auto &succ : it->second) {
        // don't add duplicates
        if (onePredCompleted.end() ==
            std::find(onePredCompleted.begin(), onePredCompleted.end(), succ)) {
          onePredCompleted.push_back(succ);
        }
      }
    }
  }

  {
    std::stringstream ss;
    ss << "one pred completed: ";
    for (const auto &op : onePredCompleted) {
      ss << op->desc() << ",";
    }
    STDERR(ss.str());
  }

  STDERR("reject ops already done or with incomplete preds...");
  std::vector<std::shared_ptr<OpBase>> candidates;
  for (const auto &cOp : onePredCompleted) {
    // reject ops that we've already done
    if (completed.contains_unbound(cOp)) {
      if (!quiet)
        STDERR(cOp->name() << " already done");
      continue;
    }

    // reject ops that all preds are not done
    bool allPredsCompleted = true;
    for (const auto &pred : g.preds_.at(cOp)) {
      if (!completed.contains_unbound(pred)) {
        STDERR(cOp->desc() << " missing pred " << pred->desc());
        allPredsCompleted = false;
        break;
      }
    }
    if (!allPredsCompleted) {
      STDERR(cOp->name() << " missing a pred");
      continue;
    }
    candidates.push_back(cOp);
  }

  {
    std::stringstream ss;
    ss << "preds complete AND not done: ";
    for (const auto &op : candidates) {
      ss << op->desc() << ",";
    }
    STDERR(ss.str());
  }

  // create all possible platform assignments of the graph frontier
  std::vector<std::shared_ptr<BoundOp>> result;
  for (const std::shared_ptr<OpBase> &candidate : candidates) {
    std::vector<std::shared_ptr<BoundOp>> bounds = make_platform_variations(plat, candidate);
    result.insert(result.end(), bounds.begin(), bounds.end());
  }

  return result;
}

/*
(2)
return a copy of g with an unbound version of op replaced with op
*/
Graph<OpBase> mcts::bind_unbound_vertex(const Graph<OpBase> &g,
                                        const std::shared_ptr<BoundOp> &op) {
  Graph<OpBase> gp = g; // g'
  if (!gp.contains(op)) {
    if (auto bgo = std::dynamic_pointer_cast<BoundGpuOp>(op)) {
      STDERR("replace " << bgo->unbound() << " with " << op->desc());
      gp.replace(bgo->unbound(), op);
    }
  }

  if (!gp.contains(op)) {
    THROW_RUNTIME("");
  }
  return gp;
}

/*
(3)
considering the `completed` so far, the graph, and the platform,
return all synchronizations that are needed before op can actually be
appended to the completed
return {} if none needed
*/
std::vector<std::shared_ptr<BoundOp>>
mcts::get_syncs_before_op(const Graph<OpBase> &g,
                          const Sequence<BoundOp> &completed,
                          const std::shared_ptr<BoundOp> &op) {
  typedef EventSynchronizer Synchronizer;

  std::vector<std::shared_ptr<BoundOp>> syncs;

  if (Synchronizer::is_synced(op, g, completed)) {
    STDERR(op->desc() << " is synced");
  } else { // otherwise synchronizers should be added
    STDERR(op->desc() << " is not synced with preds");
    syncs = Synchronizer::make_syncs(op, g, completed, true);
    {
      std::stringstream ss;
      ss << "generated synchronizers for " << op->desc() << ":";
      for (const auto &sync : syncs) {
        ss << sync->desc() << ", ";
      }
      STDERR(ss.str());
    }
  }
  return syncs;
}