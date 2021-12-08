#include "sched/ops_cuda.hpp"

#include "sched/macro_at.hpp"

#include <iostream>

void CudaEventRecord::update_name(
    const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &preds, 
    const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &succs
) {
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

std::string CudaEventRecord::desc() const { 
    std::stringstream ss;
    ss << "{" << name() << ", e:" << event_ << "}";
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

void CudaStreamWaitEvent::update_name(
    const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &preds, 
    const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &succs
) {
    std::stringstream ss;
    ss << "CudaStreamWaitEvent";
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
    CUDA_RUNTIME(cudaStreamWaitEvent(plat.cuda_stream(stream_), plat.cuda_event(event_), 0 /*flags*/));
}


void CudaEventSync::update_name(
    const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &preds, 
    const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &succs
) {
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

void CudaEventSync::run(Platform &plat ) {
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

void StreamWait::update_name(
    const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &preds, 
    const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &succs
) {
    std::stringstream ss;
    ss << "StreamWait";
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

void StreamSync::run(Platform &plat)
{
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

void StreamSync::update_name(
    const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &preds, 
    const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &succs
) {
    std::stringstream ss;
    ss << "StreamSync";
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

nlohmann::json BoundGpuOp::json() const { 
    nlohmann::json j = op_->json();
    j["stream"] = stream();
    return j;
}
