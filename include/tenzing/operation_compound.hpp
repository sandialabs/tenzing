#pragma once

#include "tenzing/operation.hpp"
#include "tenzing/graph.hpp"

/*! \brief not executable, represents a graph of suboperations
*/
class CompoundOp : public OpBase {
public:
    CompoundOp(const Graph<OpBase> &_graph) : graph(_graph) {}
    Graph<OpBase> graph;
};