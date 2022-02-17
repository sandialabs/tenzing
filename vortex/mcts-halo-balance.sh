#!/bin/bash
#BSUB -J mcts-halo-balance
#BSUB -o mcts-halo-balance.o%J
#BSUB -e mcts-halo-balance.e%J
#BSUB -W 2:00
#BSUB -nnodes 1

# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.

DIR=/ascldap/users/cwpears/repos/sched
OUT=$DIR/vortex/mcts-halo-balance.csv
EXE=$DIR/build-vortex/src_mcts_halo/mcts-halo-balance

source $DIR/load-env.sh

date
jsrun --smpiargs="-gpu" -n 4 -g 1 -c 1 -r 4 -l gpu-gpu,gpu-cpu -b rs $EXE | tee $OUT
date
