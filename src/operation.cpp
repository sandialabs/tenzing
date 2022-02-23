/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "sched/operation.hpp"
#include "sched/cuda/ops_cuda.hpp"
#include "sched/macro_at.hpp"

#include <sstream>

nlohmann::json OpBase::json() const {
  nlohmann::json j;
  j["name"] = name();
  return j;
}

nlohmann::json NoOp::json() const {
  nlohmann::json j;
  j["name"] = name();
  j["kind"] = "NoOp";
  return j;
}

void keep_uniques(std::vector<std::shared_ptr<BoundOp>> &v) {
  for (size_t i = 0; i < v.size(); ++i) {
    for (size_t j = i + 1; j < v.size(); ++j) {
      if (v[i]->eq(v[j])) {
        v.erase(v.begin() + j);
        --j;
      }
    }
  }
}

std::vector<std::shared_ptr<BoundOp>> make_platform_variations(const Platform &plat,
                                                               const std::shared_ptr<OpBase> &op) {
  std::vector<std::shared_ptr<BoundOp>> ret;
  if (auto gpuOp = std::dynamic_pointer_cast<GpuOp>(op)) {
    for (const auto &stream : plat.streams_) {
      ret.push_back(std::make_shared<BoundGpuOp>(gpuOp, stream.id_));
    }
  } else if (auto cpuOp = std::dynamic_pointer_cast<BoundOp>(op)) {
    ret.push_back(cpuOp);
  } else {
    THROW_RUNTIME("unexpected kind of op when generating platform variations");
  }
  return ret;
}
