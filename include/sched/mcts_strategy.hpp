#pragma once

namespace mcts {

/* base class for strategy context and state
   provides a no-op ostream output overload
*/
struct StrategyContext {};
inline std::ostream &operator<<(std::ostream &os, const StrategyContext &) {return os;}

struct StrategyState {};
inline std::ostream &operator<<(std::ostream &os, const StrategyState &) {return os;}

} // namespace mcts