#!/bin/bash
#BSUB -J spmv-mcts-coverage
#BSUB -o spmv-mcts-coverage.o%J
#BSUB -e spmv-mcts-coverage.e%J
#BSUB -W 2:00
#BSUB -nnodes 1

# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.

DIR=/ascldap/users/cwpears/repos/sched
OUT=$DIR/vortex/spmv-mcts-coverage.csv
EXE=$DIR/build-vortex/src_spmv/spmv-mcts-coverage

source $DIR/load-env.sh

date
jsrun --smpiargs="-gpu" -n 4 -g 1 -c 2 -r 4 -l gpu-gpu,gpu-cpu -b packed:1 $EXE | tee $OUT
date
