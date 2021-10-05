#include "run_strategy.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int err = doit<mcts::AntiCorrelation>(argc, argv);
    MPI_Finalize();
}

