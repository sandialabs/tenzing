/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include <string>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "cuda/cuda_runtime.hpp"
#include "platform.hpp"

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
    virtual std::unique_ptr<OpBase> clone() override { \
        return std::unique_ptr<OpBase>(static_cast<OpBase *>(new TYPE(*this))); \
    }

#define LT_DEF(TYPE) \
    virtual bool lt(const std::shared_ptr<OpBase> &rhs) const { \
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
    virtual bool eq(const std::shared_ptr<OpBase> &rhs) const { \
        auto rp = std::dynamic_pointer_cast<const TYPE>(rhs);\
        if (!rp) return false;\
        else return *this == *rp;\
    }

class OpBase
{
public:
    virtual ~OpBase(){};
    virtual std::string name() const = 0;
    virtual std::string desc() const { return name(); }
    virtual nlohmann::json json() const;
    virtual std::unique_ptr<OpBase> clone() = 0;
    virtual bool eq(const std::shared_ptr<OpBase> &rhs) const = 0;
    virtual bool lt(const std::shared_ptr<OpBase> &rhs) const = 0;
    virtual int tag() const  {
        return typeid(*this).hash_code();
    }

    // for map compare
    struct compare_lt {
        bool operator()(const std::shared_ptr<OpBase> &a, const std::shared_ptr<OpBase> &b) const {
            bool aLtB = a->lt(b);
            // STDERR(a->name() << " < " << b->name() << " = " << aLtB);
            return aLtB;
        }
    };
};

/*! \brief not executable, represents multiple implementation choices for an operation
*/
class ChoiceOp : public OpBase {
    virtual std::vector<std::shared_ptr<OpBase>> choices() const = 0;
};

class Expander;
/*! \brief not executable, represents a graph of suboperations
*/
class CompoundOp : public OpBase {
    virtual Expander &expander() const = 0;
};



class BoundOp : public OpBase {
public:
    virtual void run(Platform &/*plat*/) = 0;
};


class CpuOp : public BoundOp
{};



// keep unique entries in v
void keep_uniques(std::vector<std::shared_ptr<BoundOp>> &v);

/* a wrapper that turns a Gpu node into a CPU node
   by running it in a specific stream
*/

class Start : public CpuOp
{
public:
    std::string name() const override { return "start"; }
    EQ_DEF(Start);
    LT_DEF(Start);
    CLONE_DEF(Start);
    bool operator<(const Start &rhs) const {(void)rhs; return false; }
    bool operator==(const Start &rhs) const {(void)rhs; return true; }
    virtual void run(Platform &/*plat*/) override {};
};

class End : public CpuOp
{
public:
    std::string name() const override { return "end"; }
    EQ_DEF(End);
    LT_DEF(End);
    CLONE_DEF(End);
    bool operator<(const End &/*rhs*/) const {return false; }
    bool operator==(const End &/*rhs*/) const {return true; }
    virtual void run(Platform &/*plat*/) override {}
};


/* a node that does nothing
*/
class NoOp: public CpuOp {
    std::string name_;
public:
    NoOp(const std::string &name) : name_(name) {}
    std::string name() const override { return name_; }
    nlohmann::json json() const override;
    EQ_DEF(NoOp);
    LT_DEF(NoOp);
    CLONE_DEF(NoOp);
    bool operator<(const NoOp &rhs) const {
        return name_ < rhs.name_;
    }
    bool operator==(const NoOp &rhs) const {
        return name_ == rhs.name_;
    }
    virtual void run(Platform &/*plat*/) override {};
};


// produce the various places on the platform the op can run.
// e.g. if op is a GpuNode, then it could be assigned to multiple streams
std::vector<std::shared_ptr<BoundOp>> make_platform_variations(
    const Platform &plat,
    const std::shared_ptr<OpBase> &op
);



