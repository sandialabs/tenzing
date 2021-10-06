#include "spmv_stream_mcts.cuh"
#include "sched/mcts_strategy_random.hpp"

int main(int argc, char **argv)
{
    return doit<mcts::Random>(argc, argv);
}
