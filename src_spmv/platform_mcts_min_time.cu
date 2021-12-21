#include "sched/mcts_strategy_min_time.hpp"
#include "platform_mcts.cuh"

int main(int argc, char **argv)
{
    mcts::Opts opts;
    opts.dumpTreePrefix = "spmv_min_time";
    return platform_mcts<mcts::MinTime>(opts, argc, argv);
}
