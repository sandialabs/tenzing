#!/bin/bash
#BSUB -J spmv-comparison
#BSUB -o spmv-comparison.o%J
#BSUB -e spmv-comparison.e%J
#BSUB -W 2:00
#BSUB -nnodes 1

DIR=/ascldap/users/cwpears/repos/sched


source $DIR/load-env.sh

OUT=$DIR/vortex/spmv-mcts-balancehist.csv
EXE=$DIR/build-vortex/src_spmv/spmv-mcts-balancehist
date
jsrun --smpiargs="-gpu" -n 4 -g 1 -c 1 -r 4 -l gpu-gpu,gpu-cpu -b rs $EXE | tee $OUT
date

OUT=$DIR/vortex/spmv-mcts-normanticorr.csv
EXE=$DIR/build-vortex/src_spmv/spmv-mcts-normanticorr
date
jsrun --smpiargs="-gpu" -n 4 -g 1 -c 1 -r 4 -l gpu-gpu,gpu-cpu -b rs $EXE | tee $OUT
date

OUT=$DIR/vortex/spmv-mcts-normrootcorr.csv
EXE=$DIR/build-vortex/src_spmv/spmv-mcts-normrootcorr
date
jsrun --smpiargs="-gpu" -n 4 -g 1 -c 1 -r 4 -l gpu-gpu,gpu-cpu -b rs $EXE | tee $OUT
date

OUT=$DIR/vortex/spmv-mcts-random.csv
EXE=$DIR/build-vortex/src_spmv/spmv-mcts-random
date
jsrun --smpiargs="-gpu" -n 4 -g 1 -c 1 -r 4 -l gpu-gpu,gpu-cpu -b rs $EXE | tee $OUT
date


# comprehensive search of all schedules
OUT=$DIR/vortex/spmv-stream-brute.csv
EXE=$DIR/build-vortex/src_spmv/spmv-stream-brute
date
jsrun --smpiargs="-gpu" -n 4 -g 1 -c 1 -r 4 -l gpu-gpu,gpu-cpu -b rs $EXE | tee $OUT
date