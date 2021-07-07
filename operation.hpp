#pragma once

#include <string>
#include <iostream>
#include <memory>

#include "cuda_runtime.h"

#define __CLASS__ std::remove_reference<decltype(classMacroImpl(this))>::type
template<class T> T& classMacroImpl(const T* t);

class Node {
public:
    virtual ~Node(){};
    virtual std::string name() { return "<anon>"; }
    virtual std::unique_ptr<Node> clone() = 0;
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
    virtual std::unique_ptr<Node> clone() override {return std::unique_ptr<Node>(static_cast<Node*>(new Start(*this)));}
};

class End : public CpuNode
{
public:
    std::string name() override { return "End"; }
    virtual std::unique_ptr<Node> clone() override {return std::unique_ptr<Node>(static_cast<Node*>(new End(*this)));}
};

/* cause waiter to wait on current state of waitee
   this node can be inserted by the scheduler when GPU operations
   in different streams are ordered

   TODO: could decouple these calls in the future?
*/
class StreamWait : public CpuNode{
    cudaEvent_t event_;
    cudaStream_t waitee_, waiter_;
    public:
    StreamWait(cudaStream_t waitee, cudaStream_t waiter) : waitee_(waitee), waiter_(waiter) {
        CUDA_RUNTIME(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    }
    ~StreamWait() {/* FIXME: stream cleanup */ }

    std::string name() override { return std::string("StreamWait(") + std::to_string(uintptr_t(waitee_)) + "," + std::to_string(uintptr_t(waiter_)) + ")"; }
    virtual void run() override {
        CUDA_RUNTIME(cudaEventRecord(event_, waitee_));
        CUDA_RUNTIME(cudaStreamWaitEvent(waiter_, event_, 0 /*flags*/));
    }

    virtual std::unique_ptr<Node> clone() override {return std::unique_ptr<Node>(static_cast<Node*>(new __CLASS__(*this)));}
};

class StreamSync : public CpuNode
{
    cudaStream_t stream_;
public:

    StreamSync(cudaStream_t stream) : stream_(stream) {}
    std::string name() override { return std::string("StreamSync(") + std::to_string(uintptr_t(stream_)) + ")"; }
    virtual void run() override
    {
        CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    }
    virtual std::unique_ptr<Node> clone() override {return std::unique_ptr<Node>(static_cast<Node*>(new __CLASS__(*this)));}
};

/* an operation that executes on a stream
*/
class GpuNode : public Node {
public:
    virtual void run(cudaStream_t) {}
};

/* a wrapper that turns a Gpu node into a CPU node
   by running it in a specific stream
*/
class StreamedOp : public CpuNode {
    std::unique_ptr<GpuNode> node_; // the operation
    cudaStream_t stream_; // the stream this operation will be in

public:
    StreamedOp(std::unique_ptr<GpuNode> node, cudaStream_t stream) : node_(std::move(node)), stream_(stream) {}
    StreamedOp(const StreamedOp &other) : stream_(other.stream_) {
        // we know for sure we are cloning a GpuNode
        GpuNode *p = static_cast<GpuNode*>(other.node_->clone().release());
        node_ = std::move(std::unique_ptr<GpuNode>(p));
    }
    virtual void run() { node_->run(stream_); }
    std::string name() override { return node_->name() + "(" + std::to_string(uintptr_t(stream_)) + ")"; }


    virtual std::unique_ptr<Node> clone() {
        return std::unique_ptr<Node>(static_cast<Node*>(new StreamedOp(*this)));
    }

    cudaStream_t stream() { return stream_; }
};

#undef __CLASS__