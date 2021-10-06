#include "spmv_stream_mcts.cuh"
#include "sched/mcts_strategy_unvisited.hpp"

int main(int argc, char **argv)
{
    return doit<mcts::Unvisited>(argc, argv);
}
