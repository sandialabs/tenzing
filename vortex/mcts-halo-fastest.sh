#!/bin/bash
#BSUB -J mcts-halo-fastest
#BSUB -o mcts-halo-fastest.o%J
#BSUB -e mcts-halo-fastest.e%J
#BSUB -W 2:00
#BSUB -nnodes 1

DIR=/ascldap/users/cwpears/repos/sched
OUT=$DIR/vortex/mcts-halo-fastest.csv
EXE=$DIR/build-vortex/src_mcts_halo/mcts-halo-fastest

source $DIR/load-env.sh

date
jsrun --smpiargs="-gpu" -n 4 -g 1 -c 1 -r 4 -l gpu-gpu,gpu-cpu -b rs $EXE | tee mcts-halo-fastest.csv
date
