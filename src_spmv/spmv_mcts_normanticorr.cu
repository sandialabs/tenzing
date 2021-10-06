#include "spmv_stream_mcts.cuh"

int main(int argc, char **argv)
{
    return doit<mcts::NormalizedAntiCorrelation>(argc, argv);
}
