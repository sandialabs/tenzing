/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "halo_run_strategy.hpp"
#include "tenzing/mcts_strategy_balance_hist.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int err = doit<mcts::BalanceHistogram>(argc, argv);
    MPI_Finalize();
}

