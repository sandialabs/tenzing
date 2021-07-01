#pragma once

#include <string>
#include <set>
#include <iostream>

#include <cuda_runtime.h>

class Node {
public:
    std::set<Node *> succs;
    std::set<Node *> preds;

    // do op after this
    Node *then(Node *op)
    {
        succs.insert(op);
        op->preds.insert(this);
        return op;
    }

    virtual ~Node(){};
    virtual std::string name() { return "<anon>"; }
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
};

class End : public CpuNode
{
public:
    std::string name() override { return "End"; }
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
template<typename T>
class StreamedOp : public CpuNode {
    T node_; // the operation
    cudaStream_t stream_; // the stream this operation will be in

    StreamedOp(const T &node, cudaStream_t stream) : node_(node), stream_(stream) {}
    virtual void run() { node_.run(stream_); }
    bool operator==(const StreamedOp &rhs) {
        return node_ == rhs.node_ && stream_ == rhs.stream_;
    }

    virtual Node *clone() {return new StreamedOp(*this); }
};