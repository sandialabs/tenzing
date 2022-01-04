#include "sched/mcts_strategy_fast_min.hpp"
#include "platform_mcts.cuh"

int main(int argc, char **argv)
{
    mcts::Opts opts;
    opts.dumpTreePrefix = "spmv_fast_min_";
    return platform_mcts<mcts::FastMin>(opts, argc, argv);
}
