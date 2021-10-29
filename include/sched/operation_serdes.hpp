#pragma once

#include <nlohmann/json.hpp>

#include "operation.hpp"
#include "ops_cuda.hpp"
#include "graph.hpp"


void to_json(nlohmann::json& j, const std::shared_ptr<OpBase>& sp, const Graph<OpBase> &g) {
    j = sp->json();
    j["in_graph"] = g.contains(sp);
}

template <typename T>
void to_json(nlohmann::json& j, const std::vector<T>& v, const Graph<OpBase> &g) {
    j = nlohmann::json::array({});
    for (const auto &e : v) {
        nlohmann::json je;
        to_json(je, e, g);
        j.insert(j.end(), je);
    }
}

/* this is not the type signature for nlohmann::json from_json, use must be a bit explicit
   FIXME: GpuOp is in here also? What if we compile without CUDA?
*/
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
        THROW_RUNTIME("name not found in graph");
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



template <typename T>
void from_json(const nlohmann::json& j, const Graph<OpBase> &g, std::vector<T>& v) {

    v.clear();
    for (const auto &e : j) {
        T t;
        from_json(e, g, t);
        v.push_back(t);
    }
}