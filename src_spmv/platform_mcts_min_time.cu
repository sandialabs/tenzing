/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "sched/mcts_strategy_fast_min.hpp"
#include "platform_mcts.cuh"

int main(int argc, char **argv)
{
    mcts::Opts opts;
    opts.dumpTreePrefix = "spmv_fast_min_";
    return platform_mcts<mcts::FastMin>(opts, argc, argv);
}
