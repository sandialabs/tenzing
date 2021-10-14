#include "sched/mcts_strategy_fastest.hpp"

#include "run_strategy.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int err = doit<mcts::PreferFastest>(argc, argv);
    MPI_Finalize();
}

