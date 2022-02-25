/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "tenzing/cuda/ops_cuda.hpp"

#include "tenzing/macro_at.hpp"

#include <iostream>

void CudaEventRecord::update_name(const std::set<std::shared_ptr<OpBase>> &preds,
                                  const std::set<std::shared_ptr<OpBase>> &succs) {
  std::stringstream ss;
  ss << "CER";
  if (!preds.empty()) {
    ss << "-after";
    for (const auto &e : preds) {
      ss << "-" << e->name();
    }
  }

  if (!succs.empty()) {
    ss << "-b4";
    for (const auto &e : succs) {
      ss << "-" << e->name();
    }
  }

  name_ = ss.str();
}

std::string CudaEventRecord::desc() const {
  std::stringstream ss;
  ss << "{" << name() << ", e:" << event_ << ", s:" << stream_ << "}";
  return ss.str();
}

nlohmann::json CudaEventRecord::json() const {
  nlohmann::json j;
  j["name"] = name();
  j["stream"] = stream();
  j["event"] = event();
  j["kind"] = "CudaEventRecord";
  return j;
}

void CudaEventRecord::run(Platform &plat) {
  CUDA_RUNTIME(cudaEventRecord(plat.cuda_event(event_), plat.cuda_stream(stream_)));
}

void CudaStreamWaitEvent::update_name(const std::set<std::shared_ptr<OpBase>> &preds,
                                      const std::set<std::shared_ptr<OpBase>> &succs) {
  std::stringstream ss;
  ss << "CSWE";
  if (!preds.empty()) {
    ss << "-after";
    for (const auto &e : preds) {
      ss << "-" << e->name();
    }
  }

  if (!succs.empty()) {
    ss << "-b4";
    for (const auto &e : succs) {
      ss << "-" << e->name();
    }
  }

  name_ = ss.str();
}

std::string CudaStreamWaitEvent::desc() const {
  std::stringstream ss;
  ss << "{" << name() << ", s:" << stream_ << ", e:" << event_ << "}";
  return ss.str();
}

nlohmann::json CudaStreamWaitEvent::json() const {
  nlohmann::json j;
  j["name"] = name();
  j["stream"] = stream();
  j["event"] = event();
  j["kind"] = "CudaStreamWaitEvent";
  return j;
}

void CudaStreamWaitEvent::run(Platform &plat) {
  CUDA_RUNTIME(
      cudaStreamWaitEvent(plat.cuda_stream(stream_), plat.cuda_event(event_), 0 /*flags*/));
}

void CudaEventSync::update_name(const std::set<std::shared_ptr<OpBase>> &preds,
                                const std::set<std::shared_ptr<OpBase>> &succs) {
  std::stringstream ss;
  ss << "CES";
  if (!preds.empty()) {
    ss << "-after";
    for (const auto &e : preds) {
      ss << "-" << e->name();
    }
  }

  if (!succs.empty()) {
    ss << "-b4";
    for (const auto &e : succs) {
      ss << "-" << e->name();
    }
  }

  name_ = ss.str();
}

std::string CudaEventSync::desc() const {
  std::stringstream ss;
  ss << "{" << name() << ", e:" << event_ << "}";
  return ss.str();
}

nlohmann::json CudaEventSync::json() const {
  nlohmann::json j;
  j["name"] = name();
  j["event"] = event();
  j["kind"] = "CudaEventSync";
  return j;
}

void CudaEventSync::run(Platform &plat) {
  CUDA_RUNTIME(cudaEventSynchronize(plat.cuda_event(event_)));
}

nlohmann::json StreamWait::json() const {
  nlohmann::json j;
  j["name"] = name();
  j["waiter"] = waiter();
  j["waitee"] = waitee();
  j["kind"] = "StreamWait";
  return j;
}

void StreamWait::update_name(const std::set<std::shared_ptr<OpBase>> &preds,
                             const std::set<std::shared_ptr<OpBase>> &succs) {
  std::stringstream ss;
  ss << "StreamWait";
  if (!preds.empty()) {
    ss << "-after";
    for (const auto &e : preds) {
      ss << "-" << e->name();
    }
  }

  if (!succs.empty()) {
    ss << "-b4";
    for (const auto &e : succs) {
      ss << "-" << e->name();
    }
  }

  name_ = ss.str();
}

void StreamSync::run(Platform &plat) {
  cudaError_t err = cudaStreamSynchronize(plat.cuda_stream(stream_));
  if (cudaSuccess != err) {
    THROW_RUNTIME("CUDA error in " << name());
  }
  CUDA_RUNTIME(err);
}

nlohmann::json StreamSync::json() const {
  nlohmann::json j;
  j["name"] = name();
  j["stream"] = stream();
  j["kind"] = "StreamSync";
  return j;
}

void StreamSync::update_name(const std::set<std::shared_ptr<OpBase>> &preds,
                             const std::set<std::shared_ptr<OpBase>> &succs) {
  std::stringstream ss;
  ss << "StreamSync";
  if (!preds.empty()) {
    ss << "-after";
    for (const auto &e : preds) {
      ss << "-" << e->name();
    }
  }

  if (!succs.empty()) {
    ss << "-b4";
    for (const auto &e : succs) {
      ss << "-" << e->name();
    }
  }

  name_ = ss.str();
}

nlohmann::json BoundGpuOp::json() const {
  nlohmann::json j = op_->json();
  j["stream"] = stream();
  return j;
}

std::vector<Stream> BoundGpuOp::get_streams() const { return {stream_}; }



void from_json(const nlohmann::json& j, std::shared_ptr<CudaEventRecord> &op) {
    Event event;
    Stream stream;
    std::string name;
    j.at("event").get_to(event);
    j.at("stream").get_to(stream);
    j.at("name").get_to(name);
    op = std::make_shared<CudaEventRecord>(event, stream, name);
}

void from_json(const nlohmann::json& j, std::shared_ptr<CudaStreamWaitEvent> &op) {
    Event event;
    Stream stream;
    std::string name;
    j.at("event").get_to(event);
    j.at("stream").get_to(stream);
    j.at("name").get_to(name);
    op = std::make_shared<CudaStreamWaitEvent>(stream, event, name);
}

void from_json(const nlohmann::json& j, std::shared_ptr<CudaEventSync> &op) {
    Event event;
    std::string name;
    j.at("event").get_to(event);
    j.at("name").get_to(name);
    op = std::make_shared<CudaEventSync>(event, name);
}