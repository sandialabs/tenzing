#include "sched/mcts_strategy_coverage.hpp"
#include "platform_mcts.cuh"

int main(int argc, char **argv)
{
    mcts::Opts opts;
    opts.dumpTreePrefix = "spmv_coverage_";
    return platform_mcts<mcts::Coverage>(opts, argc, argv);
}
