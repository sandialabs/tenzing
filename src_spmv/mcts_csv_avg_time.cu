#include "mcts_csv.cuh"
#include "sched/mcts_strategy_avg_time.hpp"

int main(int argc, char **argv)
{
    return mcts_csv<mcts::AvgTime>(argc, argv);
}
