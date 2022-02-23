#include "sched/state.hpp"

#include "sched/decision.hpp"

State State::apply(const Decision &d) {

    try {
        const ThenOp &to = dynamic_cast<const ThenOp&>(d);
        State ret = *this;
        ret.sequence_.push_back(to.op());
        return ret;
    } catch (std::bad_cast) {
        // pass
    }

    try {
        const ExpandOp &eo = dynamic_cast<const ExpandOp&>(d);
        State ret = *this;
        #warning skeleton when applying ExpandOp
        // get preds and succs of op
        std::vector<std::shared_ptr<OpBase>> succs, preds;
        ret.graph_ = eo.expander()(eo.op, ret.graph_, preds, succs);
        return ret;
    } catch (std::bad_cast) {
        // pass
    }

    try {
        const AssignOpStream &aos = dynamic_cast<const AssignOpStream&>(d);
        #warning skeleton when aplpying AssignOpStream
    } catch (std::bad_cast) {
        // pass
    }

    try {
        const ChooseOp &co = dynamic_cast<const ChooseOp&>(d);
        #warning skeleton when aplpying ChooseOp
    } catch (std::bad_cast) {
        // pass
    }

    THROW_RUNTIME("failed to apply decision, unexpected Decision type");

}

std::vector<State> State::frontier(Platform &plat, bool quiet) {

  /*
    find all vertices in the graph that are not completed, but all preds are completed
  */
  {
    std::stringstream ss;
    ss << "frontier for state: ";
    for (const auto &op : sequence_) {
      ss << op->desc() << ",";
    }
    STDERR(ss.str());
  }

  // all successors of the sequence have at least one pred completed
  STDERR("consider ops with >= 1 pred completed...");
  std::vector<std::shared_ptr<OpBase>> onePredCompleted;
  for (const auto &cOp : sequence_) {
    if (!quiet)
      STDERR("...succs of " << cOp->desc() << " (@" << cOp.get() << ")");

    // some nodes in the path will not be in the graph (inserted syncs)
    // other nodes in the path are bound versions of that in the graph

    auto it = graph_.succs_find_or_find_unbound(cOp);
    if (graph_.succs_.end() != it) {

      // all successors of a completed op have at least one pred completed
      for (const auto &succ : it->second) {
        // don't add duplicates
        if (onePredCompleted.end() ==
            std::find(onePredCompleted.begin(), onePredCompleted.end(), succ)) {
          onePredCompleted.push_back(succ);
        }
      }
    }
  }

  {
    std::stringstream ss;
    ss << "one pred completed: ";
    for (const auto &op : onePredCompleted) {
      ss << op->desc() << ",";
    }
    STDERR(ss.str());
  }

  STDERR("reject ops already done or with incomplete preds...");
  std::vector<std::shared_ptr<OpBase>> candidates;
  for (const auto &cOp : onePredCompleted) {
    // reject ops that we've already done
    if (sequence_.contains_unbound(cOp)) {
      if (!quiet)
        STDERR(cOp->name() << " already done");
      continue;
    }

    // reject ops that all preds are not done
    bool allPredsCompleted = true;
    for (const auto &pred : graph_.preds_.at(cOp)) {
      if (!sequence_.contains_unbound(pred)) {
        STDERR(cOp->desc() << " missing pred " << pred->desc());
        allPredsCompleted = false;
        break;
      }
    }
    if (!allPredsCompleted) {
      STDERR(cOp->name() << " missing a pred");
      continue;
    }
    candidates.push_back(cOp);
  }

  {
    std::stringstream ss;
    ss << "preds complete AND not done: ";
    for (const auto &op : candidates) {
      ss << op->desc() << ",";
    }
    STDERR(ss.str());
  }


  // get all possible Decisions that can be made from this state
  std::vector<std::shared_ptr<Decision>> decisions = get_decisions(candidates);


  // apply decisions to the state
  std::vector<State> result;
  for (const Decision &decision : decisions) {
      State state = apply(decision);
      result.push_back(state);
  }

  // remove duplicate states
  #warning unimplemented state deduplication

  return result;

};