#include "mcts_csv.cuh"
#include "sched/mcts_strategy_random.hpp"

int main(int argc, char **argv)
{
    return mcts_csv<mcts::Random>(argc, argv);
}
