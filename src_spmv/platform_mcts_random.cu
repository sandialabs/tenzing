#include "sched/mcts_strategy_random.hpp"
#include "platform_mcts.cuh"

int main(int argc, char **argv)
{
    return platform_mcts<mcts::Random>(argc, argv);
}
