#pragma once

/*! \brief Represents a specific transition between states

    Inheriters don't really do anything, they just hold data necessary for their specific decision
*/
class Decision {
    public:
    virtual ~Decision() {}
};

/*! \brief Do `op` next
 */
class ThenOp : public Decision {
public:
  ThenOp(std::shared_ptr<BoundOp> op) : op_(op) {}
  const std::shared_ptr<BoundOp> &op() const { return op_; }

private:
  std::shared_ptr<BoundOp> op_;
};

/*! \brief expands compound operation `op` in the graph using `expander`
 */
class ExpandOp : public Decision {
public:
  ExpandOp(const std::shared_ptr<CompoundOp> op) : op_(op) {}
  const std::shared_ptr<CompoundOp> &op() const { return op_; }

private:
  std::shared_ptr<CompoundOp> op_;
};

/*! \brief chooses one of the options in the ChoiceOp
 */
class ChooseOp : public Decision {
public:
  ChooseOp(const std::shared_ptr<ChoiceOp> orig, const std::shared_ptr<OpBase> replacement) : orig_(orig), replacement_(replacement) {}
  const std::shared_ptr<ChoiceOp> &orig() const { return orig_; }
  const std::shared_ptr<OpBase> &replacement() const { return replacement_; }

private:
  std::shared_ptr<ChoiceOp> orig_;
  std::shared_ptr<OpBase> replacement_;
};

/*! \brief expands compound operation `op` in the graph using `expander`
 */
class AssignOpStream : public Decision {
public:
  AssignOpStream(const std::shared_ptr<GpuOp> op, Stream stream) : op_(op) {}
  const std::shared_ptr<GpuOp> &op() const { return op_; }
  const Stream &stream() const {return stream_;}

private:
  std::shared_ptr<GpuOp> op_;
  Stream stream_;
};
