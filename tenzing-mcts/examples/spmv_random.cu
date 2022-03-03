/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "tenzing/mcts/mcts_strategy_random.hpp"
#include "spmv_run_strategy.cuh"

int main(int argc, char **argv) {
  tenzing::mcts::Opts opts;
  opts.dumpTreePrefix = "spmv_random_";
  return doit<tenzing::mcts::Random>(opts, argc, argv);
}
