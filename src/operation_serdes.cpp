/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "tenzing/operation_serdes.hpp"
#include "tenzing/operation_compound.hpp"

/* these return BoundOp because every operation in a benchmarkable sequence must be a boundop
*/
std::shared_ptr<BoundOp> recurse(const nlohmann::json &j, const std::shared_ptr<OpBase> &op);
std::shared_ptr<BoundOp> recurse(const nlohmann::json &j, const Graph<OpBase> &g);

std::shared_ptr<BoundOp> recurse(const nlohmann::json &j, const std::shared_ptr<OpBase> &op) {
  if (j.at("name") == op->name()) {
    if (auto cpuOp = std::dynamic_pointer_cast<BoundOp>(op)) {
      return cpuOp;
    } else if (auto gpuOp = std::dynamic_pointer_cast<GpuOp>(op)) {
      // create a BoundGpuOp using the stream
      Stream stream = j.at("stream");
      return std::make_shared<BoundGpuOp>(gpuOp, stream);
    } else {
      THROW_RUNTIME("op was not a BoundOp or a GpuOp ");
    }
  } else if (const auto &cmOp = std::dynamic_pointer_cast<CompoundOp>(op)) {
    return recurse(j, cmOp->graph());
  } else if (const auto &chOp = std::dynamic_pointer_cast<ChoiceOp>(op)) {
    for (const auto &choice : chOp->choices()) {
      auto found = recurse(j, choice);
      if (found) {
        return found;
      }
    }
    return nullptr; // no match in choice op
  } else if (j.contains("kind")) {
    const std::string &kind = j.at("kind");
    if ("CudaEventRecord" == kind) {
      std::shared_ptr<CudaEventRecord> bop;
      from_json(j, bop);
      return bop;
    } else if ("CudaEventSync" == kind) {
      std::shared_ptr<CudaEventSync> bop;
      from_json(j, bop);
      return bop;
    } else if ("CudaStreamWaitEvent" == kind) {
      std::shared_ptr<CudaStreamWaitEvent> bop;
      from_json(j, bop);
      return bop;
    } else {
      THROW_RUNTIME("unexpected operation kind '" << kind << "' for operation missing from graph "
                                                  << j.dump());
    }
  } else {
    return nullptr; // did not find a match
  }
}

std::shared_ptr<BoundOp> recurse(const nlohmann::json &j, const Graph<OpBase> &g) {

  for (const auto &kv : g.succs_) {
    auto found = recurse(j, kv.first);
    if (found) {
      return found;
    }
  }
  return nullptr;
}

void from_json(const nlohmann::json &j, const Graph<OpBase> &g, std::shared_ptr<BoundOp> &n) {
  auto needle = recurse(j, g);
  if (!needle) {
    THROW_RUNTIME("failure to deserialize " << j);
  } else {
    n = needle;
  }
}
