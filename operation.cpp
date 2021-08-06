#include "operation.hpp"

#include <sstream>

#if 0
bool Op::operator<(const Op &rhs) const {
    return ptr_->lt(rhs.ptr_);
}
bool Op::operator==(const Op &rhs) const {
    return ptr_->eq(rhs.ptr_);
}
#endif


void StreamSync::update_name(const std::set<std::shared_ptr<Node>, Node::compare_lt> &preds, const std::set<std::shared_ptr<Node>, Node::compare_lt> &succs) {
    std::stringstream ss;
    ss << "StreamSync";
    ss << "-after";
    for (const auto &e : preds) {
        ss << "-" << e->name();
    }
    ss << "-b4";
    for (const auto &e : succs) {
        ss << "-" << e->name();
    }

    name_ = ss.str();
}

void StreamWait::update_name(const std::set<std::shared_ptr<Node>, Node::compare_lt> &preds, const std::set<std::shared_ptr<Node>, Node::compare_lt> &succs) {
    std::stringstream ss;
    ss << "StreamWait";
    ss << "-after";
    for (const auto &e : preds) {
        ss << "-" << e->name();
    }
    ss << "-b4";
    for (const auto &e : succs) {
        ss << "-" << e->name();
    }

    name_ = ss.str();
}