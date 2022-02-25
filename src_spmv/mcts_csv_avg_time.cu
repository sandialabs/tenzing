/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "mcts_csv.cuh"
#include "tenzing/mcts_strategy_avg_time.hpp"

int main(int argc, char **argv)
{
    mcts::Opts opts;
    opts.dumpTreePrefix = "spmv_csv_avg_time_";
    return mcts_csv<mcts::AvgTime>(opts, argc, argv);
}
