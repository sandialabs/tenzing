#include "mcts_csv.cuh"
#include "sched/mcts_strategy_coverage.hpp"

int main(int argc, char **argv)
{
    return mcts_csv<mcts::Coverage>(argc, argv);
}
