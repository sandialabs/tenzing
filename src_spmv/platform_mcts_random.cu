#include "sched/mcts_strategy_random.hpp"
#include "platform_mcts.cuh"

int main(int argc, char **argv)
{
    mcts::Opts opts;
    opts.dumpTreePrefix = "spmv_random_";
    return platform_mcts<mcts::Random>(opts, argc, argv);
}
