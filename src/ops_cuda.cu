#include "sched/ops_cuda.hpp"

#include "sched/macro_at.hpp"

#include <iostream>

void CudaEventRecord::update_name(const std::set<std::shared_ptr<Node>, Node::compare_lt> &preds, const std::set<std::shared_ptr<Node>, Node::compare_lt> &succs) {
    std::stringstream ss;
    ss << "CudaEventRecord";
    ss << "-after";
    for (const auto &e : preds) {
        ss << "-" << e->name();
    }
    ss << "-b4";
    for (const auto &e : succs) {
        ss << "-" << e->name();
    }

    name_ = ss.str();
}

std::string CudaEventRecord::json() const { 
    std::stringstream ss;
    ss << "{";
    ss << "name: \"" << name() << "\""; 
    ss << ", stream: " << stream_; 
    ss << "}";
    return ss.str();
}

void CudaEventRecord::run() {
    CUDA_RUNTIME(cudaEventRecord(event_, stream_));
}



void CudaEventSync::update_name(const std::set<std::shared_ptr<Node>, Node::compare_lt> &preds, const std::set<std::shared_ptr<Node>, Node::compare_lt> &succs) {
    std::stringstream ss;
    ss << "CudaEventSync";
    ss << "-after";
    for (const auto &e : preds) {
        ss << "-" << e->name();
    }
    ss << "-b4";
    for (const auto &e : succs) {
        ss << "-" << e->name();
    }

    name_ = ss.str();
}

void CudaEventSync::run() {
    CUDA_RUNTIME(cudaEventSynchronize(event_));
}