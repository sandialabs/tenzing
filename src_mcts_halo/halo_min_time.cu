#include "sched/mcts_strategy_fast_min.hpp"

#include "run_strategy.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int err = doit<mcts::FastMin>(argc, argv);
    MPI_Finalize();
}

