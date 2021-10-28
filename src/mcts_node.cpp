#include "sched/mcts_node.hpp"

#include "sched/operation.hpp"
#include "sched/ops_cuda.hpp"

// true iff e in v
bool contains(
    const std::vector<std::shared_ptr<BoundOp>> &v, 
    const std::shared_ptr<OpBase> &e
) {
    for (const auto &ve : v) {
        if (ve->unbound() == e.get()) {
            return true;
        }
    }
    return false;
}

// uses cudaEventRecord, cudaStreamWait and cudaEventSync for synchronization
struct EventSynchronizer {

    // find a in v or return v.end()
    static std::vector<std::shared_ptr<BoundOp>>::const_iterator find(
        const std::vector<std::shared_ptr<BoundOp>> &v,
        const std::shared_ptr<OpBase> &a
    ) {
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (a->eq(*it)) {
                return it;
            }
        }
        return v.end();    
    }

    /* true if a -> cudaEventRecord -> cudaEventSync -> b
    */
    static bool is_synced_gpu_then_cpu(
        const std::shared_ptr<BoundGpuOp> &a,
        const std::shared_ptr<CpuOp> &b,
        const std::vector<std::shared_ptr<BoundOp>> &path
    ) {
        // find a
        auto ai = find(path, a);
        if (path.end() == ai) {
            THROW_RUNTIME("counldn't find a " << a->name() << " in path");
        }
        auto bi = find(path, b);
        if (path.end() == bi) {
            THROW_RUNTIME("counldn't find b " << b->name() << " in path");
        }
        if (ai >= bi) {
            THROW_RUNTIME("a " << a->name() << " is not before b " << b ->name());
        }

        std::shared_ptr<CudaEventRecord> cer;
        for (auto it = ai; it != bi; ++it) {
            if (auto p = std::dynamic_pointer_cast<CudaEventRecord>(*it)) {
                if (p->stream() == a->stream()) {
                    cer = p;
                }
            } else if (auto ces = std::dynamic_pointer_cast<CudaEventSync>(*it)) {
                if (cer) {
                    if (ces->event() == cer->event()) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // true if a -> cudaEventRecord -> cudaStreamWaitEvent -> b
    static bool is_synced_gpu_then_gpu(
        const std::shared_ptr<BoundGpuOp> &a,
        const std::shared_ptr<BoundGpuOp> &b,
        const std::vector<std::shared_ptr<BoundOp>> &path
    ) {
        // find a
        auto ai = find(path, a);
        if (path.end() == ai) {
            THROW_RUNTIME("counldn't find a " << a->name() << " in path");
        }
        auto bi = find(path, b);
        if (path.end() == bi) {
            THROW_RUNTIME("counldn't find b " << b->name() << " in path");
        }
        if (ai >= bi) {
            THROW_RUNTIME("a " << a->name() << " is not before b " << b ->name());
        }

        std::shared_ptr<CudaEventRecord> cer;
        for (auto it = ai; it != bi; ++it) {
            if (auto p = std::dynamic_pointer_cast<CudaEventRecord>(*it)) {
                if (p->stream() == a->stream()) {
                    cer = p;
                }
            } else if (auto cswe = std::dynamic_pointer_cast<CudaStreamWaitEvent>(*it)) {
                if (cer) {
                    if (cswe->event() == cer->event() && cswe->stream() == b->stream()) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // return cudaEventRecord or cudaEventSync as needed
    static std::shared_ptr<BoundOp> make_sync_gpu_then_cpu(
        const std::shared_ptr<BoundGpuOp> &a,
        const std::shared_ptr<CpuOp> &b,
        const std::vector<std::shared_ptr<BoundOp>> &path
    ) {
#warning skeleton
    }

    // return cudaEventRecord or cudaStreamWait as needed
    static std::shared_ptr<BoundOp> make_sync_gpu_then_gpu(
        const std::shared_ptr<BoundGpuOp> &a,
        const std::shared_ptr<BoundGpuOp> &b,
        const std::vector<std::shared_ptr<BoundOp>> &path
    ) {
        // find a
        auto ai = find(path, a);
        if (path.end() == ai) {
            THROW_RUNTIME("counldn't find a " << a->name() << " in path");
        }
        auto bi = find(path, b);
        if (path.end() == bi) {
            THROW_RUNTIME("counldn't find b " << b->name() << " in path");
        }
        if (ai >= bi) {
            THROW_RUNTIME("a " << a->name() << " is not before b " << b ->name());
        }

        std::shared_ptr<CudaEventRecord> cer;
        for (auto it = ai; it != bi; ++it) {
            if (auto p = std::dynamic_pointer_cast<CudaEventRecord>(*it)) {
                if (p->stream() == a->stream()) {
                    cer = p;
                    break;
                }
            } 
        }
        if (cer) {
            // assume there is no CudaStreamWaitEvent, so produce one that synces b's stream with 
            return std::make_shared<CudaStreamWaitEvent>(b->stream(), cer->event());
        } else {
            return std::make_shared<CudaEventRecord>(a->stream());
        }
    }

public:
    // true iff bo were after path and is synced with any preds in path
    static bool is_synced(
        const std::shared_ptr<BoundOp> &bo, 
        const Graph<OpBase> &g,
        const std::vector<std::shared_ptr<BoundOp>> &path
    ) {
        // find all ops on path that are predecessors of bo
        for (const auto &pred : g.preds_.at(bo)) {

            // various CPU/GPU sync combinations
            auto bc = std::dynamic_pointer_cast<CpuOp>(bo);
            auto bg = std::dynamic_pointer_cast<BoundGpuOp>(bo);
            auto pc = std::dynamic_pointer_cast<CpuOp>(pred);
            auto pg = std::dynamic_pointer_cast<BoundGpuOp>(pred);

            if (pc && bc) { // cpu -> cpu (nothing)
                ; // no sync needed
            } else if (pg && bc) { // gpu -> cpu (CER & CEW)
                if (!is_synced_gpu_then_cpu(pg, bc, path)) {
                    return false;
                }
            } else if (pc && bg) { // cpu -> gpu
                ; // no sync needed
            } else if (pg && bg) { // gpu -> gpu (maybe CER & CSW)
                if (!is_synced_gpu_then_gpu(pg, bg, path)) {
                    return false;
                }
            } else {
                THROW_RUNTIME("unpected Op combination");
            }
        }
        return true;
    }

    // return any operations to insert after `path` that would help synchronize `bo` with its predecessors
    static std::vector<std::shared_ptr<BoundOp>> make_syncs(
        const std::shared_ptr<BoundOp> &bo, 
        const Graph<OpBase> &g,
        const std::vector<std::shared_ptr<BoundOp>> &path
    ) {
        std::vector<std::shared_ptr<BoundOp>> syncs;

        // find all ops on path that are predecessors of bo
        for (const auto &pred : g.preds_.at(bo)) {

            // various CPU/GPU sync combinations
            auto bc = std::dynamic_pointer_cast<CpuOp>(bo);
            auto bg = std::dynamic_pointer_cast<BoundGpuOp>(bo);
            auto pc = std::dynamic_pointer_cast<CpuOp>(pred);
            auto pg = std::dynamic_pointer_cast<BoundGpuOp>(pred);

            if (pc && bc) { // cpu -> cpu (nothing)
                ; // no sync needed
            } else if (pg && bc) { // gpu -> cpu (CER & CEW)
                if (!is_synced_gpu_then_cpu(pg, bc, path)) {
                    syncs.push_back(make_sync_gpu_then_cpu(pg, bc, path));
                }
            } else if (pc && bg) { // cpu -> gpu
                ; // no sync needed
            } else if (pg && bg) { // gpu -> gpu (maybe CER & CSW)
                if (!is_synced_gpu_then_gpu(pg, bg, path)) {
                    syncs.push_back(make_sync_gpu_then_gpu(pg, bg, path));
                }
            } else {
                THROW_RUNTIME("unpected Op combination");
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

*/
std::vector<std::shared_ptr<BoundOp>> mcts::get_frontier(
    const Platform &plat,
    const Graph<OpBase> &g,
    const std::vector<std::shared_ptr<BoundOp>> &completed
) {

    typedef EventSynchronizer Synchronizer;

    /* 
    find candidate operations for the frontier
        all predecessors in `completed`
        is not itself in `completed`
    */
    std::vector<std::shared_ptr<OpBase>> candidates;

    std::vector<std::shared_ptr<OpBase>> onePredCompleted;
    for (const auto &cOp : completed) {
        for (const auto &succ : g.succs_.at(cOp)) {
            onePredCompleted.push_back(succ);
        }
    }

    for (const auto &cOp : onePredCompleted) {
        // reject ops that we've already done
        if (contains(completed, cOp)) {
            continue;
        }

        // reject ops that all preds are not done
        bool allPredsCompleted = true;
        for (const auto &pred : g.preds_.at(cOp)) {
            if (!contains(completed, pred)) {
                allPredsCompleted = false;
                break;
            }
        }
        if (!allPredsCompleted) {
            continue;
        }
        candidates.push_back(cOp);
    }

    std::vector<std::shared_ptr<BoundOp>> frontier;

    // candidates may or may not be assigned to resources
    // get the viable resource assignments for the candidates
    for (const auto &candidate : candidates) {
        std::vector<std::shared_ptr<BoundOp>> bounds = make_platform_variations(plat, candidate);


        for (const std::shared_ptr<BoundOp> &bound : bounds) {
            // if the candidate is already synchronized with its preds, it can be added to the frontier
            if (Synchronizer::is_synced(bound, g, completed)) {
                frontier.push_back(bound);
            } else { // otherwise synchronizers should be added instead
                std::vector<std::shared_ptr<BoundOp>> syncs = Synchronizer::make_syncs(bound, g, completed);
                for (const auto &sync : syncs) {
                    frontier.push_back(sync);
                }
            }
        }
    }
    keep_uniques(frontier);

    return frontier;
}