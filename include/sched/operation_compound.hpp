#pragma once

#include "sched/operation.hpp"
#include "sched/graph.hpp"

/*! \brief not executable, represents a graph of suboperations
*/
class CompoundOp : public OpBase {
public:
    CompoundOp(const Graph<OpBase> &_graph) : graph(_graph) {}
    Graph<OpBase> graph;
};