#pragma once

#include <string>
#include <set>
#include <iostream>

class Operation
{
public:
    virtual void run() { }

    std::set<Operation *> succs;
    std::set<Operation *> preds;

    // do op after this
    Operation *then(Operation *op)
    {
        succs.insert(op);
        op->preds.insert(this);
        return op;
    }

    virtual ~Operation(){};
    virtual std::string name() { return "<anon>"; }
};

class Start : public Operation
{
public:
    std::string name() override { return "Start"; }
};

class End : public Operation
{
public:
    std::string name() override { return "End"; }
};