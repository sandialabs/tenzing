#include "mcts_csv.cuh"
#include "sched/mcts_strategy_min_time.hpp"

int main(int argc, char **argv)
{
    mcts::Opts opts;
    opts.dumpTreePrefix = "spmv_csv_min_time";
    return mcts_csv<mcts::MinTime>(opts, argc, argv);
}
