#pragma once

#include <memory>

#include "operation.hpp"
#include "operation_compound.hpp"
#include "cuda/ops_cuda.hpp"

/*! \brief Represents a specific transition between states

    Basically just a tagged union
*/
class Decision {
    public:
    virtual ~Decision() {}
};

/*! \brief Do `op` next
 */
class ExecuteOp : public Decision {
public:
  ExecuteOp(const std::shared_ptr<BoundOp> &_op) : op(_op) {}
  std::shared_ptr<BoundOp> op;
};

/*! \brief expands compound operation `op` in the graph using `expander`
 */
class ExpandOp : public Decision {
public:
  ExpandOp(const std::shared_ptr<CompoundOp> &_op) : op(_op) {}
  std::shared_ptr<CompoundOp> op;
};

/*! \brief chooses one of the options in the ChoiceOp
 */
class ChooseOp : public Decision {
public:
  ChooseOp(const std::shared_ptr<ChoiceOp> &_orig, const std::shared_ptr<OpBase> &_replacement) : orig(_orig), replacement(_replacement) {}
  std::shared_ptr<ChoiceOp> orig;
  std::shared_ptr<OpBase> replacement;
};

/*! \brief expands compound operation `op` in the graph using `expander`
 */
class AssignOpStream : public Decision {
public:
  AssignOpStream(const std::shared_ptr<GpuOp> &_op, Stream _stream) : op(_op), stream(_stream) {}
  std::shared_ptr<GpuOp> op;
  Stream stream;
};
