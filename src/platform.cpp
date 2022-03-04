/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "tenzing/platform.hpp"

void to_json(nlohmann::json &j, const Stream &s) {
  j = nlohmann::json(s.id_);
}

void from_json(const nlohmann::json &j, Stream &s) {
  j.get_to(s.id_);
}

void to_json(nlohmann::json &j, const Event &e) {
  j = nlohmann::json(e.id_);
}

void from_json(const nlohmann::json &j, Event &e) {
  j.get_to(e.id_);
}

bool ResourceMap::contains(const Event &event) const {
  const bool result = events_.count(event);
  return result;
}

bool ResourceMap::insert(const Event &event, cudaEvent_t cevent) {
  return events_.insert(std::make_pair(event, cevent)).second;
}

cudaEvent_t CudaEventPool::new_event() {
  while (i_ >= events_.size()) {
    cudaEvent_t e;
    CUDA_RUNTIME(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    events_.push_back(e);
  }
  return events_[i_++];
}

int Platform::num_streams() const {
  if (streams_.size() != cStreams_.size()) {
    THROW_RUNTIME("internal platform stream bookkeeping error");
  }
  return streams_.size();
}