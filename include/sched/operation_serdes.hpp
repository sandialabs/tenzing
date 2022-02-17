/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include <nlohmann/json.hpp>

#include "operation.hpp"
#include "ops_cuda.hpp"
#include "graph.hpp"

inline 
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

/* set `n` to the op in `g` that corresponds to the serialization `j`

   this is not the type signature for nlohmann::json from_json, use must be a bit explicit
   FIXME: GpuOp is in here also? What if we compile without CUDA?
*/
void from_json(const nlohmann::json& j, const Graph<OpBase> &g, std::shared_ptr<BoundOp> &n);


template <typename T>
void from_json(const nlohmann::json& j, const Graph<OpBase> &g, std::vector<T>& v) {

    v.clear();
    for (const auto &e : j) {
        T t;
        from_json(e, g, t);
        v.push_back(t);
    }
}

void from_json(const nlohmann::json& j, std::shared_ptr<CudaEventRecord> &op);
void from_json(const nlohmann::json& j, std::shared_ptr<CudaStreamWaitEvent> &op);
void from_json(const nlohmann::json& j, std::shared_ptr<CudaEventSync> &op);