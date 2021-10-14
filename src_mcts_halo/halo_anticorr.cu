#include "run_strategy.hpp"
#include "sched/mcts_strategy_norm_anti_corr.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int err = doit<mcts::NormalizedAntiCorrelation>(argc, argv);
    MPI_Finalize();
}

