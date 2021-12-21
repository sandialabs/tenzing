#include "mcts_csv.cuh"
#include "sched/mcts_strategy_avg_time.hpp"

int main(int argc, char **argv)
{
    mcts::Opts opts;
    opts.dumpTreePrefix = "spmv_csv_avg_time_";
    return mcts_csv<mcts::AvgTime>(opts, argc, argv);
}
