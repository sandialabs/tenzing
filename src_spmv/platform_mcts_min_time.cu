#include "sched/mcts_strategy_min_time.hpp"
#include "platform_mcts.cuh"

int main(int argc, char **argv)
{
    return platform_mcts<mcts::MinTime>(argc, argv);
}
