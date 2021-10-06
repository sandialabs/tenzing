#include "run_strategy.hpp"
#include "sched/mcts_strategy_coverage.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int err = doit<mcts::Coverage>(argc, argv);
    MPI_Finalize();
}

