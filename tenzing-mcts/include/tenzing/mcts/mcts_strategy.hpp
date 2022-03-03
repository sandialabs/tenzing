/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

namespace tenzing::mcts {

/* base class for strategy context and state
   provides a no-op ostream output overload
*/
struct StrategyContext {};
inline std::ostream &operator<<(std::ostream &os, const StrategyContext &) { return os; }

struct StrategyState {
  // a line that can go in a graphviz label
  std::string graphviz_label_line() const;
};
inline std::ostream &operator<<(std::ostream &os, const StrategyState &) { return os; }

std::string StrategyState::graphviz_label_line() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

} // namespace tenzing::mcts