#include "mcts_csv.cuh"
#include "sched/mcts_strategy_random.hpp"

int main(int argc, char **argv)
{
    mcts::Opts opts;
    opts.dumpTreePrefix = "spmv_csv_random";
    return mcts_csv<mcts::Random>(opts, argc, argv);
}
