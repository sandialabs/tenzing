/* mpi-specific operations
*/

#pragma once

#include "operation.hpp"

#include <mpi.h>

#include <vector>

class CudaEventRecord : public CpuNode
{
protected:
    std::string name_;
    cudaStream_t stream_;
    cudaEvent_t event_;
public:
    CudaEventRecord(cudaStream_t stream) : name_("CudaEventRecord-anon"), stream_(stream)
    {
        CUDA_RUNTIME(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    }
    // need a new event on copy so dtor doesn't go twice
    CudaEventRecord(const CudaEventRecord &other) : name_(other.name_), stream_(other.stream_) {
        CUDA_RUNTIME(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    }
    CudaEventRecord(CudaEventRecord &&other) = delete;
    ~CudaEventRecord()
    {
        CUDA_RUNTIME(cudaEventDestroy(event_));
    }
    cudaEvent_t event() const { return event_; }
    std::string name() const override { return name_; }
    virtual std::string json() const override;
    void update_name(const std::set<std::shared_ptr<Node>, Node::compare_lt> &preds, const std::set<std::shared_ptr<Node>, Node::compare_lt> &succs);

    virtual void run() override;
    virtual int tag() const override { return 10; }

    CLONE_DEF(CudaEventRecord);
    EQ_DEF(CudaEventRecord);
    LT_DEF(CudaEventRecord);
    bool operator==(const CudaEventRecord &rhs) const {
        return name() == rhs.name();
    }
    bool operator<(const CudaEventRecord &rhs) const {
        return name() < rhs.name();
    }
};

class CudaEventSync : public CpuNode
{
    std::string name_;
    cudaEvent_t event_;

public:
    CudaEventSync(cudaEvent_t event) : name_("CudaEventSync-anon"), event_(event) {}
    std::string name() const override { return name_; }
    void update_name(const std::set<std::shared_ptr<Node>, Node::compare_lt> &preds, const std::set<std::shared_ptr<Node>, Node::compare_lt> &succs);

    virtual void run() override;
    virtual int tag() const override { return 11; }

    EQ_DEF(CudaEventSync);
    LT_DEF(CudaEventSync);
    CLONE_DEF(CudaEventSync);
    bool operator<(const CudaEventSync &rhs) const {
        return name() < rhs.name(); 
    }
    bool operator==(const CudaEventSync &rhs) const {
        return name() == rhs.name();
    }
};