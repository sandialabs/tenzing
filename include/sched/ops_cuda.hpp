/* mpi-specific operations
*/

#pragma once

#include "operation.hpp"

#include <mpi.h>

#include <vector>

/*! interfaces to query an operation about what events/streams it uses
*/
class HasEvent {
public:
    virtual std::vector<Event> get_events() const = 0;
};
class HasStream {
public:
    virtual std::vector<Stream> get_streams() const = 0;
};

/* cause waiter to wait on current state of waitee
   this node can be inserted by the scheduler when GPU operations
   in different streams are ordered
*/
class StreamWait : public BoundOp, public HasEvent
{
    std::string name_;
    Event event_;
    Stream waitee_, waiter_;

public:
    StreamWait(Stream waitee, Stream waiter, Event event) 
     : name_("StreamWait-anon"), event_(event), waitee_(waitee), waiter_(waiter) {}

    // need a new event on copy so dtor doesn't go twice
    StreamWait(const StreamWait &other) = default;
    StreamWait(StreamWait &&other) = delete;

    Stream waiter() const { return waiter_; }
    Stream waitee() const { return waitee_; }

    std::string name() const override { return name_; }
    virtual nlohmann::json json() const override;
    void update_name(const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &preds, const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &succs);
    
    virtual void run(Platform &plat) override
    {
        cudaEvent_t event = plat.cuda_event(event_);
        CUDA_RUNTIME(cudaEventRecord(event, plat.cuda_stream(waitee_)));
        CUDA_RUNTIME(cudaStreamWaitEvent(plat.cuda_stream(waiter_), event, 0 /*flags*/));
    }

    virtual int tag() const override { return 3; }

    EQ_DEF(StreamWait);
    LT_DEF(StreamWait);
    CLONE_DEF(StreamWait);
    bool operator<(const StreamWait &rhs) const {
        return name() < rhs.name();
    }
    bool operator==(const StreamWait &rhs) const {
        return name() == rhs.name();
    }

    virtual std::vector<Event> get_events() const override { return {event_}; }
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
    
    virtual int tag() const override { return 4; }

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

class CudaEventRecord : public BoundOp, public HasEvent, public HasStream
{
protected:
    std::string name_;
    Event event_;
    Stream stream_;
public:
    CudaEventRecord(Event event, Stream stream, const std::string &name = "CudaEventRecord-anon") 
     : name_(name), event_(event), stream_(stream) {}

    // need a new event on copy so dtor doesn't go twice
    CudaEventRecord(const CudaEventRecord &other) = default;
    CudaEventRecord(CudaEventRecord &&other) = delete;

    Event event() const { return event_; }
    Stream stream() const { return stream_; }
    std::string name() const override { return name_; }
    std::string desc() const override;
    virtual nlohmann::json json() const override;
    void update_name(const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &preds, const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &succs);

    virtual void run(Platform &plat) override;
    virtual int tag() const override { return 5; }

    CLONE_DEF(CudaEventRecord);
    EQ_DEF(CudaEventRecord);
    LT_DEF(CudaEventRecord);
    bool operator==(const CudaEventRecord &rhs) const {
        return name() == rhs.name();
    }
    bool operator<(const CudaEventRecord &rhs) const {
        return name() < rhs.name();
    }

    std::vector<Event> get_events() const override { return {event_}; }
    std::vector<Stream> get_streams() const override{ return {stream_}; }
};



class CudaStreamWaitEvent : public BoundOp, public HasEvent, public HasStream
{
protected:
    std::string name_;
    Stream stream_;
    Event event_; // does not own event
public:
    CudaStreamWaitEvent(Stream stream, Event event, const std::string &name = "CudaStreamWaitEvent-anon")
     : name_(name), stream_(stream), event_(event) {}
    CudaStreamWaitEvent(const CudaStreamWaitEvent &other) = default;
    CudaStreamWaitEvent(CudaStreamWaitEvent &&other) = delete;

    Event event() const { return event_; }
    Stream stream() const { return stream_; }
    std::string name() const override { return name_; }
    std::string desc() const override;
    virtual nlohmann::json json() const override;
    void update_name(const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &preds, const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &succs);

    virtual void run(Platform &plat) override;
    virtual int tag() const override { return 6; }

    CLONE_DEF(CudaStreamWaitEvent);
    EQ_DEF(CudaStreamWaitEvent);
    LT_DEF(CudaStreamWaitEvent);
    bool operator==(const CudaStreamWaitEvent &rhs) const {
        return name() == rhs.name();
    }
    bool operator<(const CudaStreamWaitEvent &rhs) const {
        return name() < rhs.name();
    }

    std::vector<Event> get_events() const override{ return {event_}; }
    std::vector<Stream> get_streams() const override{ return {stream_}; }
};

class CudaEventSync : public BoundOp, public HasEvent
{
    std::string name_;
    Event event_;

public:
    CudaEventSync(Event event, const std::string &name = "CudaEventSync-anon") : name_(name), event_(event) {}
    Event event() const { return event_; }
    std::string name() const override { return name_; }
    std::string desc() const override;
    virtual nlohmann::json json() const override;
    void update_name(const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &preds, const std::set<std::shared_ptr<OpBase>, OpBase::compare_lt> &succs);

    virtual void run(Platform &plat) override;
    virtual int tag() const override { return 7; }

    EQ_DEF(CudaEventSync);
    LT_DEF(CudaEventSync);
    CLONE_DEF(CudaEventSync);
    bool operator<(const CudaEventSync &rhs) const {
        return name() < rhs.name(); 
    }
    bool operator==(const CudaEventSync &rhs) const {
        return name() == rhs.name();
    }

    std::vector<Event> get_events() const override { return {event_}; }
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
class BoundGpuOp : public BoundOp, public HasStream
{
    std::shared_ptr<GpuOp> op_; // the operation
    Stream stream_;           // the stream this operation will be in

public:
    BoundGpuOp(const std::shared_ptr<GpuOp> &op, Stream stream) : op_(op), stream_(stream) {}
    BoundGpuOp(const BoundGpuOp &other) = default;
    virtual void run(Platform &plat) override { op_->run(plat.cuda_stream(stream_)); }
    std::string name() const override { return op_->name(); }
    std::string desc() const override { 
        std::stringstream ss;
        ss << "{" << name() << ", s:" << stream_ << "}";
        return ss.str();
    }
    nlohmann::json json() const override;
    
    const Stream &stream() const { return stream_; }
    virtual int tag() const override { return 8; }

    EQ_DEF(BoundGpuOp);
    LT_DEF(BoundGpuOp);
    CLONE_DEF(BoundGpuOp);
    bool operator<(const BoundGpuOp &rhs) const {
        if (stream_ < rhs.stream_) {
            return true;
        } else if (stream_ > rhs.stream_) {
            return false;
        } else {
            return op_->lt(rhs.op_);
        }
    }
    bool operator==(const BoundGpuOp &rhs) const {
        return (stream_ == rhs.stream_) && op_->eq(rhs.op_);
    }

    virtual std::shared_ptr<GpuOp> unbound() { return op_; }
    std::vector<Stream> get_streams() const override;
};