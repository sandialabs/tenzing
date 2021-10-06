#include "spmv_stream_mcts.cuh"
#include "sched/mcts_strategy_norm_root_corr.hpp"

int main(int argc, char **argv)
{
    return doit<mcts::NormRootCorr>(argc, argv);
}
