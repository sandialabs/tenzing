#include "sched/operation_serdes.hpp"

void from_json(const nlohmann::json& j, const Graph<OpBase> &g, std::shared_ptr<BoundOp> &n) {

    // read the operation name
    const std::string &name = j.at("name"); 

    // find the node's name in the graph
    std::shared_ptr<OpBase> needle;
    for (const auto &kv : g.succs_) {
        if (name == kv.first->name()) {
            if (!needle) {
                needle = kv.first;
            } else {
                THROW_RUNTIME("duplicate name found in graph");
            }
        }
    }
    if (!needle) {
        THROW_RUNTIME("operation name '" << name << "' not found in graph");

        const std::string &kind = j.at("kind");
        if (kind == "CudaEventRecord") {
            std::shared_ptr<CudaEventRecord> op;
            from_json(j, op);
            return;
        } else {
            THROW_RUNTIME("unexpected operation kind '" << kind << "' for operation missing from graph");
        }
    }

    if (auto cpuNode = std::dynamic_pointer_cast<BoundOp>(needle)) {
        n = cpuNode;
        return;
    } else if (auto gpuNode = std::dynamic_pointer_cast<GpuOp>(needle)) {
        // create a BoundGpuOp using the stream
        Stream stream = j.at("stream");
        n = std::make_shared<BoundGpuOp>(gpuNode, stream);
        return;
    } else {
        THROW_RUNTIME("unexpected node kind in from_json");
    }
}