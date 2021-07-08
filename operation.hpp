#pragma once

#include <string>
#include <iostream>
#include <memory>

#include "cuda_runtime.h"

#define __CLASS__ std::remove_reference<decltype(classMacroImpl(this))>::type
template <class T>
T &classMacroImpl(const T *t);

#define EQUAL_DEF_1(D)                                     \
    virtual bool equal(std::shared_ptr<Node> rhs) const override \
    {                                                      \
        if (auto p = std::dynamic_pointer_cast<D>(rhs))

#define EQUAL_DEF_2 \
    return false;   \
    }

class Node
{
public:
    virtual ~Node(){};
    virtual std::string name() { return "<anon>"; }
    virtual std::unique_ptr<Node> clone() = 0;
    virtual bool equal(std::shared_ptr<Node> rhs) const = 0;
    virtual int tag() const = 0; // unique per node type
};

class CpuNode : public Node
{
public:
    virtual void run() {}
};

class Start : public CpuNode
{
public:
    std::string name() override { return "Start"; }
    EQUAL_DEF_1(Start)
    {
        return true;
    }
    EQUAL_DEF_2
    virtual std::unique_ptr<Node> clone() override { return std::unique_ptr<Node>(static_cast<Node *>(new Start(*this))); }
    virtual int tag() const override { return 0; }
};

class End : public CpuNode
{
public:
    std::string name() override { return "End"; }
    EQUAL_DEF_1(End)
    {
        return true;
    }
    EQUAL_DEF_2
    virtual std::unique_ptr<Node> clone() override { return std::unique_ptr<Node>(static_cast<Node *>(new End(*this))); }
    virtual int tag() const override { return 1; }
};

/* cause waiter to wait on current state of waitee
   this node can be inserted by the scheduler when GPU operations
   in different streams are ordered

   TODO: could decouple these calls in the future?
*/
class StreamWait : public CpuNode
{
    cudaEvent_t event_;
    cudaStream_t waitee_, waiter_;

public:
    StreamWait(cudaStream_t waitee, cudaStream_t waiter) : waitee_(waitee), waiter_(waiter)
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
    std::string name() override { return std::string("StreamWait(") + std::to_string(uintptr_t(waitee_)) + "/" + std::to_string(uintptr_t(waiter_)) + ")"; }
    virtual void run() override
    {
        CUDA_RUNTIME(cudaEventRecord(event_, waitee_));
        CUDA_RUNTIME(cudaStreamWaitEvent(waiter_, event_, 0 /*flags*/));
    }

    EQUAL_DEF_1(StreamWait)
    {
        return true;
    }
    EQUAL_DEF_2

    virtual std::unique_ptr<Node> clone() override { return std::unique_ptr<Node>(static_cast<Node *>(new __CLASS__(*this))); }
    virtual int tag() const override { return 2; }
};

class StreamSync : public CpuNode
{
    cudaStream_t stream_;

public:
    StreamSync(cudaStream_t stream) : stream_(stream) {}
    cudaStream_t stream() const { return stream_; }
    std::string name() override { return std::string("StreamSync(") + std::to_string(uintptr_t(stream_)) + ")"; }
    EQUAL_DEF_1(StreamSync)
    {
        return true;
    }
    EQUAL_DEF_2
    virtual void run() override
    {
        CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    }
    virtual std::unique_ptr<Node> clone() override { return std::unique_ptr<Node>(static_cast<Node *>(new __CLASS__(*this))); }
    virtual int tag() const override { return 3; }
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
    std::string name() override { return node_->name() + "(" + std::to_string(uintptr_t(stream_)) + ")"; }

    virtual bool equal(std::shared_ptr<Node> rhs) const {
        if (auto p = std::dynamic_pointer_cast<StreamedOp>(rhs)) {
            return node_->equal(p->node_);
        }
        return false;
    }

    virtual std::unique_ptr<Node> clone()
    {
        return std::unique_ptr<Node>(static_cast<Node *>(new StreamedOp(*this)));
    }

    cudaStream_t stream() { return stream_; }
    virtual int tag() const override { return node_->tag(); }
};

#undef __CLASS__