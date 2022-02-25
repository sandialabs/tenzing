#include "sched/state.hpp"

namespace SDP {

std::vector<std::shared_ptr<BoundOp>> State::get_syncs_before_op(const std::shared_ptr<BoundOp> &op) const {
  std::vector<std::shared_ptr<BoundOp>> syncs;

  if (Synchronizer::is_synced(op, graph_, sequence_)) {
    STDERR(op->desc() << " is synced");
  } else { // otherwise synchronizers should be added
    STDERR(op->desc() << " is not synced with preds");
    syncs = Synchronizer::make_syncs(op, graph_, sequence_, true);
    {
      std::stringstream ss;
      ss << "generated synchronizers for " << op->desc() << ":";
      for (const auto &sync : syncs) {
        ss << sync->desc() << ", ";
      }
      STDERR(ss.str());
    }
  }
  return syncs;
}

std::vector<std::shared_ptr<Decision>> State::get_decisions(Platform &plat) const {

  // find all nodes in graph that are available
  std::vector<std::shared_ptr<OpBase>> frontier = graph_.frontier(sequence_.vector());


  std::vector<std::shared_ptr<Decision>> decisions;

  for (const auto &op : frontier) {

    // any BoundOp that are available to actually execute (or a synchronization thereof)
    if (auto bop = std::dynamic_pointer_cast<BoundOp>(op)) {

      // see if the op requires synchronization
      std::vector<std::shared_ptr<BoundOp>> syncs = get_syncs_before_op(bop);

      // if not, this op is available
      if (syncs.empty()) {
        decisions.push_back(std::make_shared<ExecuteOp>(bop));
      } else { // otherwise, a synchronization of this op should be available
        for (const std::shared_ptr<BoundOp> &sync : syncs) {
          decisions.push_back(std::make_shared<ExecuteOp>(sync));
        }
      }
    }
    // any GpuOp that can be assigned to a stream
    else if (auto gop = std::dynamic_pointer_cast<GpuOp>(op)) {
      for (const Stream stream : plat.streams_) {
        decisions.push_back(std::make_shared<AssignOpStream>(gop, stream));
      }
    }
    // any CompoundOp that can be expanded
    else if (auto cop1 = std::dynamic_pointer_cast<CompoundOp>(op)) {
      decisions.push_back(std::make_shared<ExpandOp>(cop1));
    }
    // and ChoiceOp that can be chosen
    else if (auto cop2 = std::dynamic_pointer_cast<ChoiceOp>(op)) {
      for (const auto &choice : cop2->choices()) {
        decisions.push_back(std::make_shared<ChooseOp>(cop2, choice));
      }
    }
  }

  return decisions;
}

State State::apply(const Decision &d) const {

  try {
    const ExecuteOp &to = dynamic_cast<const ExecuteOp &>(d);
    State ret = *this;
    ret.sequence_.push_back(to.op);
    return ret;
  } catch (std::bad_cast) {
    // pass
  }

  try {
    const ExpandOp &eo = dynamic_cast<const ExpandOp &>(d);
    return State(graph_.clone_but_expand(eo.op, eo.op->graph), sequence_);
  } catch (std::bad_cast) {
    // pass
  }

  try {
    const AssignOpStream &aos = dynamic_cast<const AssignOpStream &>(d);
    State ret = *this;
    ret.graph_ = graph_.clone_but_replace(std::make_shared<BoundGpuOp>(aos.op, aos.stream), aos.op);
    return ret;
  } catch (std::bad_cast) {
    // pass
  }

  try {
    const ChooseOp &co = dynamic_cast<const ChooseOp &>(d);
    return State(graph_.clone_but_replace(co.replacement, co.orig), sequence_);
  } catch (std::bad_cast) {
    // pass
  }

  THROW_RUNTIME("failed to apply decision, unexpected Decision type");
}

std::vector<State> State::frontier(Platform &plat, bool quiet) {

  // get all possible Decisions that can be made from this state
  std::vector<std::shared_ptr<Decision>> decisions = get_decisions(plat);

  // apply decisions to the state
  std::vector<State> result;
  for (const auto &decision : decisions) {
    State state = apply(*decision);
    result.push_back(state);
  }

// remove duplicate states
#warning unimplemented state deduplication

  return result;
};

} // namespace SDP