/* mpi-specific operations
*/

#pragma once

#include "operation.hpp"

#include <mpi.h>

#include <vector>

/* cause waiter to wait on current state of waitee
   this node can be inserted by the scheduler when GPU operations
   in different streams are ordered
*/
class StreamWait : public BoundOp
{
    std::string name_;
    cudaEvent_t event_;
    Stream waitee_, waiter_;

public:
    StreamWait(Stream waitee, Stream waiter) : name_("StreamWait-anon"), waitee_(waitee), waiter_(waiter)
    {
        CUDA_RUNTIME(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    }
    // need a new event on copy so dtor doesn't go twice
    StreamWait(const StreamWait &other) : waitee_(other.waitee_), waiter_(other.waiter_) {
        CUDA_RUNTIME(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    }
    StreamWait(StreamWait &&other) = delete;

    ~StreamWait()
    {
        CUDA_RUNTIME(cudaEventDestroy(event_));
    }

    Stream waiter() const { return waiter_; }
    Stream waitee() const { return waitee_; }
    std::string name() const override { return name_; }
    virtual nlohmann::json json() const override;
    void update_name(const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &preds, const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &succs);
    
    virtual void run(Platform &plat) override
    {
        CUDA_RUNTIME(cudaEventRecord(event_, plat.cuda_stream(waitee_)));
        CUDA_RUNTIME(cudaStreamWaitEvent(plat.cuda_stream(waiter_), event_, 0 /*flags*/));
    }

    virtual int tag() const override { return 2; }

    EQ_DEF(StreamWait);
    LT_DEF(StreamWait);
    CLONE_DEF(StreamWait);
    bool operator<(const StreamWait &rhs) const {
        return name() < rhs.name();
    }
    bool operator==(const StreamWait &rhs) const {
        return name() == rhs.name();
    }
};

class StreamSync : public BoundOp
{
    std::string name_;
    Stream stream_;

public:
    StreamSync(Stream stream) : name_("streamsync-anon"), stream_(stream) {}
    Stream stream() const { return stream_; }
    std::string name() const override { return name_; }
    nlohmann::json json() const override;
    void update_name(const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &preds, const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &succs);

    virtual void run(Platform &plat) override;
    
    virtual int tag() const override { return 3; }

    EQ_DEF(StreamSync);
    LT_DEF(StreamSync);
    CLONE_DEF(StreamSync);
    bool operator<(const StreamSync &rhs) const {
        return name() < rhs.name(); 
    }
    bool operator==(const StreamSync &rhs) const {
        (void) rhs; return true;
    }
};

class CudaEventRecord : public BoundOp
{
protected:
    std::string name_;
    Stream stream_;
    cudaEvent_t event_;
public:
    CudaEventRecord(Stream stream) : name_("CudaEventRecord-anon"), stream_(stream)
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
    Stream stream() const { return stream_; }
    std::string name() const override { return name_; }
    virtual nlohmann::json json() const override;
    void update_name(const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &preds, const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &succs);

    virtual void run(Platform &plat) override;
    virtual int tag() const override { return 11; }

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

class CudaStreamWaitEvent : public BoundOp
{
protected:
    std::string name_;
    Stream stream_;
    cudaEvent_t event_; // does not own event
public:
    CudaStreamWaitEvent(Stream stream, cudaEvent_t event)
     : name_("CudaStreamWaitEvent-anon"), stream_(stream), event_(event) {}
    CudaStreamWaitEvent(const CudaStreamWaitEvent &other) = default;
    CudaStreamWaitEvent(CudaStreamWaitEvent &&other) = delete;

    cudaEvent_t event() const { return event_; }
    Stream stream() const { return stream_; }
    std::string name() const override { return name_; }
    virtual nlohmann::json json() const override;
    void update_name(const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &preds, const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &succs);

    virtual void run(Platform &plat) override;
    virtual int tag() const override { return 12; }

    CLONE_DEF(CudaStreamWaitEvent);
    EQ_DEF(CudaStreamWaitEvent);
    LT_DEF(CudaStreamWaitEvent);
    bool operator==(const CudaStreamWaitEvent &rhs) const {
        return name() == rhs.name();
    }
    bool operator<(const CudaStreamWaitEvent &rhs) const {
        return name() < rhs.name();
    }
};

class CudaEventSync : public BoundOp
{
    std::string name_;
    cudaEvent_t event_;

public:
    CudaEventSync(cudaEvent_t event) : name_("CudaEventSync-anon"), event_(event) {}
    cudaEvent_t event() const { return event_; }
    std::string name() const override { return name_; }
    void update_name(const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &preds, const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &succs);

    virtual void run(Platform &plat) override;
    virtual int tag() const override { return 12; }

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

/* an operation that executes on a stream
*/
class GpuOp : public OpBase
{
public:
    virtual void run(cudaStream_t) = 0;
};


/* a wrapper that turns a Gpu node into a CPU node
   by running it in a specific stream
*/
class BoundGpuOp : public BoundOp
{
    std::shared_ptr<GpuOp> op_; // the operation
    Stream stream_;           // the stream this operation will be in

public:
    BoundGpuOp(const std::shared_ptr<GpuOp> &op, Stream stream) : op_(op), stream_(stream) {}
    BoundGpuOp(const BoundGpuOp &other) = default;
    virtual void run(Platform &plat) override { op_->run(plat.cuda_stream(stream_)); }
    std::string name() const override { return op_->name(); }
    nlohmann::json json() const override;
    
    const Stream &stream() const { return stream_; }
    virtual int tag() const override { return 4; }

    EQ_DEF(BoundGpuOp);
    LT_DEF(BoundGpuOp);
    CLONE_DEF(BoundGpuOp);
    bool operator<(const BoundGpuOp &rhs) const {
        return op_->lt(rhs.op_);
    }
    bool operator==(const BoundGpuOp &rhs) const {
        return op_->eq(rhs.op_);
    }

    virtual OpBase *unbound() override { return op_.get(); }
};