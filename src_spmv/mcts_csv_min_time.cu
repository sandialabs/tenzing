#include "mcts_csv.cuh"
#include "sched/mcts_strategy_min_time.hpp"

int main(int argc, char **argv)
{
    return mcts_csv<mcts::MinTime>(argc, argv);
}
