/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "tenzing/operation_serdes.hpp"

void from_json(const nlohmann::json& j, const Graph<OpBase> &g, std::shared_ptr<BoundOp> &n) {
    // read the operation name
    const std::string &name = j.at("name"); 

    // find the node's name in the graph
    #warning json in_graph field
    #warning recursive descent into CompoundOp
    #warning check all ChoiceOp 
    std::shared_ptr<OpBase> needle;
    for (const auto &kv : g.succs_) {
        if (name == kv.first->name()) {
            if (!needle) {
                needle = kv.first;
            } else {
                THROW_RUNTIME("duplicate name found in graph " << j.dump());
            }
        }
    }
    if (!needle) {
        const std::string &kind = j.at("kind");
        if ("CudaEventRecord" == kind) {
            std::shared_ptr<CudaEventRecord> op;
            from_json(j, op);
            n = op;
            goto check_and_return;
        } else if ("CudaEventSync" == kind) {
            std::shared_ptr<CudaEventSync> op;
            from_json(j, op);
            n = op;
            goto check_and_return;
        } else if ("CudaStreamWaitEvent" == kind) {
            std::shared_ptr<CudaStreamWaitEvent> op;
            from_json(j, op);
            n = op;
            goto check_and_return;
        } else {
            THROW_RUNTIME("unexpected operation kind '" << kind << "' for operation missing from graph " << j.dump());
        }
    }

    if (auto cpuNode = std::dynamic_pointer_cast<BoundOp>(needle)) {
        n = cpuNode;
        goto check_and_return;
    } else if (auto gpuNode = std::dynamic_pointer_cast<GpuOp>(needle)) {
        // create a BoundGpuOp using the stream
        Stream stream = j.at("stream");
        n = std::make_shared<BoundGpuOp>(gpuNode, stream);
        goto check_and_return;
    } else {
        THROW_RUNTIME("unexpected node kind in from_json");
    }

check_and_return: // oof
    if (!n) {
        THROW_RUNTIME("unexpected failure to deserialize " << j.dump());
    }
    return;
}

