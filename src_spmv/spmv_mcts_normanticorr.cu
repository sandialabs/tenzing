#include "spmv_stream_mcts.cuh"
#include "sched/mcts_strategy_norm_anti_corr.hpp"

int main(int argc, char **argv)
{
    return doit<mcts::NormalizedAntiCorrelation>(argc, argv);
}
