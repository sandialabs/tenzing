#include "spmv_stream_mcts.cuh"
#include "sched/mcts_strategy_coverage.hpp"

int main(int argc, char **argv)
{
    return doit<mcts::Coverage>(argc, argv);
}
