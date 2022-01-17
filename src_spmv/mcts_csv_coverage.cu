/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "mcts_csv.cuh"
#include "sched/mcts_strategy_coverage.hpp"

int main(int argc, char **argv)
{
    mcts::Opts opts;
    opts.dumpTreePrefix = "spmv_csv_coverage_";
    return mcts_csv<mcts::Coverage>(opts, argc, argv);
}
