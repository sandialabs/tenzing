/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "tenzing/mcts/mcts_strategy_coverage.hpp"
#include "halo_run_strategy.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int err = doit<tenzing::mcts::Coverage>(argc, argv);
    MPI_Finalize();
}

