#pragma once

#include "tenzing/operation.hpp"
#include "tenzing/graph.hpp"

/*! \brief not executable, represents a graph of suboperations
*/
class CompoundOp : public OpBase {
public:

    /// \brief the graph of suboperations represented by this operation
    virtual const Graph<OpBase> &graph() const = 0;
};