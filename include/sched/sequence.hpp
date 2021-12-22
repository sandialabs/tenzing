#pragma once

#include <map>
#include <vector>
#include <memory>

#include "sched/platform.hpp"
#include "sched/operation.hpp"

template <typename OpType>
using Sequence = std::vector<std::shared_ptr<OpType>>;

struct Equivalence
{
    std::map<Stream, Stream> sMap;
    std::map<Event, Event> eMap;

    // true if equivalent
    operator bool() const
    {
        return !sMap.empty() && !eMap.empty();
    }
};

Equivalence get_equivalence(const Sequence<BoundOp> &a, const Sequence<BoundOp> &b);