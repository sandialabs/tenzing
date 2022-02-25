#pragma once

#include "operation.hpp"
#include "sequence.hpp"

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
                                     const Sequence<BoundOp> &path);

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
      auto predi = path.find_unbound(gPred);
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
      auto predi = path.find_unbound(gPred);
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