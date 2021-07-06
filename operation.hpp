#pragma once

#include <string>
#include <set>
#include <iostream>
#include <memory>

#include <cuda_runtime.h>

class Node {
public:
    std::set<std::shared_ptr<Node>> succs;
    std::set<std::shared_ptr<Node>> preds;

    virtual ~Node(){};
    virtual std::string name() { return "<anon>"; }

    // b follows a, returning b
    static std::shared_ptr<Node> then (std::shared_ptr<Node>a, std::shared_ptr<Node>b) {
        a->succs.insert(b);
        b->preds.insert(a);
        return b;
    }


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


    virtual std::unique_ptr<Node> clone() {
        return std::unique_ptr<Node>(static_cast<Node*>(new StreamedOp(*this)));
    }
};