#include "mcts_csv.cuh"
#include "sched/mcts_strategy_coverage.hpp"

int main(int argc, char **argv)
{
    mcts::Opts opts;
    opts.dumpTreePrefix = "spmv_csv_coverage_";
    return mcts_csv<mcts::Coverage>(opts, argc, argv);
}
