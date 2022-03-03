/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include <cstdint>

#include "tenzing/counters.hpp"

namespace tenzing {
namespace counters {

struct Mcts {
  ZeroInit<double> SELECT_TIME;
  ZeroInit<double> EXPAND_TIME;
  ZeroInit<double> ROLLOUT_TIME;
  ZeroInit<double> REDUNDANT_SYNC_TIME;
  ZeroInit<double> RMAP_TIME;
  ZeroInit<double> BENCHMARK_TIME;
  ZeroInit<double> BACKPROP_TIME;
};

extern Mcts mcts;

} // namespace counters
} // namespace tenzing
