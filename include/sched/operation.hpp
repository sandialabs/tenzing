#pragma once

#include <string>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>

#include "cuda_runtime.h"

/* operation eq means that the two operations
   represent the same task, but not necessarily in the same stream

   operation::lt should create a consistent ordering of operations, regardless of their stream

   therefore, if two operations do the same thing on different data, 
    * a->eq(b) == false they should not be equal
    * a->lt(b) ^ b->lt(a) == true (one should be lt the other)
    *
 
   the easiest way to ensure this is to give them different names

*/


#define CLONE_DEF(TYPE) \
    virtual std::unique_ptr<Node> clone() override { \
        return std::unique_ptr<Node>(static_cast<Node *>(new TYPE(*this))); \
    }

#define LT_DEF(TYPE) \
    virtual bool lt(const std::shared_ptr<Node> &rhs) const { \
        if (tag() < rhs->tag()) {\
            return true;\
        } else if (tag() > rhs->tag()) {\
            return false;\
        } else {\
            const auto rp = std::dynamic_pointer_cast<const TYPE>(rhs);\
            if (!rp) {\
                std::stringstream ss;\
                ss << "LT_DEF: " << name() << " <? " << rhs->name();\
                throw std::runtime_error(ss.str());\
            }\
            return *this < *rp;\
        }\
    }

#define EQ_DEF(TYPE) \
    virtual bool eq(const std::shared_ptr<Node> &rhs) const { \
        auto rp = std::dynamic_pointer_cast<const TYPE>(rhs);\
        if (!rp) return false;\
        else return *this == *rp;\
    }

class Node
{
public:
    virtual ~Node(){};
    virtual std::string name() const = 0;
    virtual std::string json() const;
    virtual std::unique_ptr<Node> clone() = 0;
    virtual bool eq(const std::shared_ptr<Node> &rhs) const = 0;
    virtual bool lt(const std::shared_ptr<Node> &rhs) const = 0;
    virtual int tag() const = 0; // unique per node type

    // for map compare
    struct compare_lt {
        bool operator()(const std::shared_ptr<Node> &a, const std::shared_ptr<Node> &b) const {
            return a->lt(b);
        }
    };
};


template <typename T>
class Op {
private:
    std::shared_ptr<T> ptr_;
public:

    template <typename U>
    Op<U> dyn_cast() {
        return std::dynamic_pointer_cast<U>(ptr_);
    }

    bool operator<(const Op &rhs) const;
    bool operator==(const Op &rhs) const;
};


class CpuNode : public Node
{
public:
    virtual void run() {}
};

class Start : public CpuNode
{
public:
    std::string name() const override { return "start"; }
    EQ_DEF(Start);
    LT_DEF(Start);
    CLONE_DEF(Start);
    virtual int tag() const override { return 0; }
    bool operator<(const Start &rhs) const {(void)rhs; return false; }
    bool operator==(const Start &rhs) const {(void)rhs; return true; }
};

class End : public CpuNode
{
public:
    std::string name() const override { return "end"; }
    virtual int tag() const override { return 1; }
    EQ_DEF(End);
    LT_DEF(End);
    CLONE_DEF(End);
    bool operator<(const End &rhs) const {(void)rhs; return false; }
    bool operator==(const End &rhs) const {(void)rhs; return true; }
};

/* cause waiter to wait on current state of waitee
   this node can be inserted by the scheduler when GPU operations
   in different streams are ordered
*/
class StreamWait : public CpuNode
{
    std::string name_;
    cudaEvent_t event_;
    cudaStream_t waitee_, waiter_;

public:
    StreamWait(cudaStream_t waitee, cudaStream_t waiter) : name_("StreamWait-anon"), waitee_(waitee), waiter_(waiter)
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

    cudaStream_t waiter() const { return waiter_; }
    cudaStream_t waitee() const { return waitee_; }
    std::string name() const override { return name_; }
    virtual std::string json() const override;
    void update_name(const std::set<std::shared_ptr<Node>, Node::compare_lt> &preds, const std::set<std::shared_ptr<Node>, Node::compare_lt> &succs);
    virtual void run() override
    {
        CUDA_RUNTIME(cudaEventRecord(event_, waitee_));
        CUDA_RUNTIME(cudaStreamWaitEvent(waiter_, event_, 0 /*flags*/));
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

class StreamSync : public CpuNode
{
    std::string name_;
    cudaStream_t stream_;

public:
    StreamSync(cudaStream_t stream) : name_("streamsync-anon"), stream_(stream) {}
    cudaStream_t stream() const { return stream_; }
    std::string name() const override { return name_; }
    std::string json() const override;
    void update_name(const std::set<std::shared_ptr<Node>, Node::compare_lt> &preds, const std::set<std::shared_ptr<Node>, Node::compare_lt> &succs);

    virtual void run() override;
    
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

/* an operation that executes on a stream
*/
class GpuNode : public Node
{
public:
    virtual void run(cudaStream_t) {}
};

/* a wrapper that turns a Gpu node into a CPU node
   by running it in a specific stream
*/
class StreamedOp : public CpuNode
{
    std::shared_ptr<GpuNode> node_; // the operation
    cudaStream_t stream_;           // the stream this operation will be in

public:
    StreamedOp(std::shared_ptr<GpuNode> node, cudaStream_t stream) : node_(std::move(node)), stream_(stream) {}
    StreamedOp(const StreamedOp &other) : stream_(other.stream_)
    {
        // we know for sure we are cloning a GpuNode
        GpuNode *p = static_cast<GpuNode *>(other.node_->clone().release());
        node_ = std::move(std::unique_ptr<GpuNode>(p));
    }
    virtual void run() { node_->run(stream_); }
    std::string name() const override { return node_->name(); }
    std::string json() const override;

    cudaStream_t stream() const { return stream_; }
    virtual int tag() const override { return 4; }

    EQ_DEF(StreamedOp);
    LT_DEF(StreamedOp);
    CLONE_DEF(StreamedOp);
    bool operator<(const StreamedOp &rhs) const {
        return node_->lt(rhs.node_);
    }
    bool operator==(const StreamedOp &rhs) const {
        return node_->eq(rhs.node_);
    }
};

