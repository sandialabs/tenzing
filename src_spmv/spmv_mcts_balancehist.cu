#include "spmv_stream_mcts.cuh"
#include "sched/mcts_strategy_balance_hist.hpp"

int main(int argc, char **argv)
{
    return doit<mcts::BalanceHistogram>(argc, argv);
}
