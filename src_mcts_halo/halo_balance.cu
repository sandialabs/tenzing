#include "run_strategy.hpp"
#include "sched/mcts_strategy_balance_hist.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int err = doit<mcts::BalanceHistogram>(argc, argv);
    MPI_Finalize();
}

