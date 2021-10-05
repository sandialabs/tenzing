#!/bin/bash
#BSUB -J mcts-halo-coverage
#BSUB -o mcts-halo-coverage.o%J
#BSUB -e mcts-halo-coverage.e%J
#BSUB -W 2:00
#BSUB -nnodes 1

DIR=/ascldap/users/cwpears/repos/sched
OUT=$DIR/vortex/mcts-halo-coverage.csv
EXE=$DIR/build-vortex/src_mcts_halo/mcts-halo-coverage

source $DIR/load-env.sh

date
jsrun --smpiargs="-gpu" -n 4 -g 1 -c 1 -r 4 -l gpu-gpu,gpu-cpu -b rs $EXE | tee $OUT
date
