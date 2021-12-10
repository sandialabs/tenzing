#include "sched/mcts_node.hpp"

#include "sched/operation.hpp"
#include "sched/ops_cuda.hpp"

// true iff unbound version of e in unbound versions of v
bool unbound_contains(
    const std::vector<std::shared_ptr<BoundOp>> &v, 
    const std::shared_ptr<OpBase> &e
) {

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

        if (uve->eq(e)) {
            return true;
        }
    }
    return false;
}

// true iff unbound version of e in unbound versions of v
std::vector<std::shared_ptr<BoundOp>>::const_iterator unbound_find(
    const std::vector<std::shared_ptr<BoundOp>> &v, 
    const std::shared_ptr<OpBase> &e
) {

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

        if (uve->eq(e)) {
            return it;
        }
    }
    return v.end();
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
        const std::shared_ptr<CpuOp> &/*b*/, // doesn't need b since b is on the CPU
        const std::vector<std::shared_ptr<BoundOp>> &path
    ) {
        // find a
        auto ai = find(path, a);
        if (path.end() == ai) {
            THROW_RUNTIME("counldn't find a " << a->name() << " in path");
        }

        std::shared_ptr<CudaEventRecord> cer;
        for (auto it = ai; it != path.end(); ++it) {
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

        // implicit sync if in same stream
        if (a->stream() == b->stream()) {
            return true;
        }

        std::shared_ptr<CudaEventRecord> cer;
        for (auto it = ai; it != path.end(); ++it) {
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

    // add whatever comes after a to produce a -> cudaEventRecord -> cudaEventSync
    static std::shared_ptr<BoundOp> make_sync_gpu_then_cpu(
        const std::shared_ptr<BoundGpuOp> &a,
        const std::shared_ptr<CpuOp> &/*b*/, // doesn't need b since b is on the CPU
        const std::vector<std::shared_ptr<BoundOp>> &path,
        Platform  &plat
    ) {
        // find a
        auto ai = find(path, a);
        if (path.end() == ai) {
            THROW_RUNTIME("counldn't find a " << a->name() << " in path");
        }

        std::shared_ptr<CudaEventRecord> cer;
        for (auto it = ai; it != path.end(); ++it) {
            if (auto p = std::dynamic_pointer_cast<CudaEventRecord>(*it)) {
                if (p->stream() == a->stream()) {
                    cer = p;
                    break;
                }
            } 
        }
        if (cer) {
            // assume there is no CudaEventSync, so produce one that synces b's stream with 
            return std::make_shared<CudaEventSync>(cer->event());
        } else {
            return std::make_shared<CudaEventRecord>(plat.new_event(), a->stream());
        }
    }

    // return cudaEventRecord or cudaStreamWait as needed
    static std::shared_ptr<BoundOp> make_sync_gpu_then_gpu(
        const std::shared_ptr<BoundGpuOp> &a,
        const std::shared_ptr<BoundGpuOp> &b,
        const std::vector<std::shared_ptr<BoundOp>> &path,
        Platform  &plat
    ) {
        // find a
        auto ai = find(path, a);
        if (path.end() == ai) {
            THROW_RUNTIME("counldn't find a " << a->name() << " in path");
        }

        if (a->stream() == b->stream()) {
            THROW_RUNTIME("make_sync_gpu_then_gpu called on ops in same stream");
        }

        std::shared_ptr<CudaEventRecord> cer;
        for (auto it = ai; it != path.end(); ++it) {
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
            return std::make_shared<CudaEventRecord>(plat.new_event(), a->stream());
        }
    }

public:
    // true iff bo were after path and is synced with any preds in path
    static bool is_synced(
        const std::shared_ptr<BoundOp> &bo, 
        const Graph<OpBase> &g,
        const std::vector<std::shared_ptr<BoundOp>> &path
    ) {

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
                THROW_RUNTIME("couldn't find " << gPred->desc() << " in path");
            }
            const std::shared_ptr<BoundOp> pred = *predi;

            STDERR("is_synced for " << pred->desc() << " ... " << bo->desc());

            // various CPU/GPU sync combinations
            // predicates are check in the graph, so they're not Bound
            auto bCpu = std::dynamic_pointer_cast<CpuOp>(bo);
            auto bGpu = std::dynamic_pointer_cast<BoundGpuOp>(bo);
            auto pCpu = std::dynamic_pointer_cast<CpuOp>(pred);
            auto pGpu = std::dynamic_pointer_cast<BoundGpuOp>(pred);
            bool pSE = std::dynamic_pointer_cast<Start>(pred) || std::dynamic_pointer_cast<End>(pred);
            bool bSE = std::dynamic_pointer_cast<Start>(bo) || std::dynamic_pointer_cast<End>(bo);

            if (pSE || bSE) { // start or end node
                ; // no sync
            } else if (pCpu && bCpu) { // cpu -> cpu (nothing)
                ; // no sync needed
            } else if (pGpu && bCpu) { // gpu -> cpu (CER & CEW)
                // auto pBound = std::dynamic_pointer_cast<BoundGpuOp>(pred);
                // if (!pBound) THROW_RUNTIME("couldn't get BoundGpuOp for " << pred->desc());
                if (!is_synced_gpu_then_cpu(pGpu, bCpu, path)) {
                    return false;
                }
            } else if (pCpu && bGpu) { // cpu -> gpu
                ; // no sync needed
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
                THROW_RUNTIME("unexpected op combination: " << pred->name() << " and " << bo->name() << ": " << ss.str());
            }
        }
        return true;
    }

    // return any operations to insert after `path` that would help synchronize `bo` with its predecessors
    static std::vector<std::shared_ptr<BoundOp>> make_syncs(
        Platform &plat,
        const std::shared_ptr<BoundOp> &bo,
        const Graph<OpBase> &g,
        const std::vector<std::shared_ptr<BoundOp>> &path
    ) {

        // graph may contain bo or the unbound version of bo
        auto it = g.preds_find_or_find_unbound(bo);
        if (it == g.preds_.end()) {
            THROW_RUNTIME("couldn't find BoundOp " << bo->name() << " in graph");
        }

        STDERR("make syncs for " << bo->desc());
        std::vector<std::shared_ptr<BoundOp>> syncs;

        // find all ops on path that are predecessors of bo
        for (const auto &gPred : it->second) {

            STDERR("pred " << gPred->desc() << " of " << bo->desc() << "...");

            // find the predecessor in the path
            auto predi = unbound_find(path, gPred);
            if (path.end() == predi) {
                THROW_RUNTIME("couldn't find " << gPred->desc() << " in path");
            }
            const std::shared_ptr<BoundOp> pred = *predi;

            // various CPU/GPU sync combinations
            auto bCpu = std::dynamic_pointer_cast<CpuOp>(bo);
            auto bGpu = std::dynamic_pointer_cast<BoundGpuOp>(bo);
            auto pCpu = std::dynamic_pointer_cast<CpuOp>(pred);
            auto pGpu = std::dynamic_pointer_cast<BoundGpuOp>(pred);
            bool pSE = std::dynamic_pointer_cast<Start>(pred) || std::dynamic_pointer_cast<End>(pred);
            bool bSE = std::dynamic_pointer_cast<Start>(bo) || std::dynamic_pointer_cast<End>(bo);

            if (pSE || bSE) { // start or end node
                ; // no sync
            } else if (pCpu && bCpu) { // cpu -> cpu (nothing)
                ; // no sync needed
            } else if (pGpu && bCpu) { // gpu -> cpu (CER & CEW)
                // auto pBound = std::dynamic_pointer_cast<BoundGpuOp>(pred);
                if (!is_synced_gpu_then_cpu(pGpu, bCpu, path)) {
                    syncs.push_back(make_sync_gpu_then_cpu(pGpu, bCpu, path, plat));
                }
            } else if (pCpu && bGpu) { // cpu -> gpu
                ; // no sync needed
            } else if (pGpu && bGpu) { // gpu -> gpu (maybe CER & CSW)
                // auto pBound = std::dynamic_pointer_cast<BoundGpuOp>(bo);
                // auto bBound = std::dynamic_pointer_cast<BoundGpuOp>(pred);
                if (!is_synced_gpu_then_gpu(pGpu, bGpu, path)) {
                    syncs.push_back(make_sync_gpu_then_gpu(pGpu, bGpu, path, plat));
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
    Platform &plat,
    const Graph<OpBase> &g,
    const std::vector<std::shared_ptr<BoundOp>> &completed
) {

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
        STDERR( "...succs of " << cOp->desc() << " (@" << cOp.get() << ")");

        // some nodes in the path will not be in the graph (inserted syncs)
        // other nodes in the path are bound versions of that in the graph

        auto it = g.succs_find_or_find_unbound(cOp);
        if (g.succs_.end() != it) {

            // all successors of a completed op have at least one pred completed
            for (const auto &succ : it->second) {
                // don't add duplicates
                if (onePredCompleted.end() == std::find(onePredCompleted.begin(), onePredCompleted.end(), succ)) {
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
        if (unbound_contains(completed, cOp)) {
            STDERR(cOp->name() << " already done");
            continue;
        }

        // reject ops that all preds are not done
        bool allPredsCompleted = true;
        for (const auto &pred : g.preds_.at(cOp)) {
            if (!unbound_contains(completed, pred)) {
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
        for (const auto &op : onePredCompleted) {
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
                STDERR("variation of " << bound->desc() << " is not synced");
                std::vector<std::shared_ptr<BoundOp>> syncs = Synchronizer::make_syncs(plat, bound, g, completed);
                STDERR("adding synchronizers to frontier:");
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