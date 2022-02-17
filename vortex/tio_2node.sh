#!/bin/bash
#BSUB -J tio_2node
#BSUB -o tio_2node.o%J
#BSUB -e tio_2node.e%J
#BSUB -W 4:00
#BSUB -nnodes 2

# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.

DIR=$HOME/repos/sched
EXE=$DIR/build-vortex/main

source $DIR/load-env.sh

date
jsrun --smpiargs="-gpu" -n 8 -g 1 -c 1 -r 4 -l gpu-gpu,gpu-cpu -b rs \
$EXE \
$HOME/heidi-thornquist-mats/tio.mtx

