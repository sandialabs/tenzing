#!/bin/bash
#BSUB -J mcts-halo-coverage
#BSUB -o mcts-halo-coverage.o%J
#BSUB -e mcts-halo-coverage.e%J
#BSUB -W 2:00
#BSUB -nnodes 1

# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.

DIR=/ascldap/users/cwpears/repos/sched
OUT=$DIR/vortex/mcts-halo-coverage.csv
EXE=$DIR/build-vortex/src_mcts_halo/mcts-halo-coverage

source $DIR/load-env.sh

date
jsrun --smpiargs="-gpu" -n 4 -g 1 -c 2 -r 4 -l gpu-gpu,gpu-cpu -b packed:1 $EXE | tee $OUT
date
