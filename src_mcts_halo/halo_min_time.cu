#include "sched/mcts_strategy_min_time.hpp"

#include "run_strategy.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int err = doit<mcts::MinTime>(argc, argv);
    MPI_Finalize();
}

