#!/bin/bash
#BSUB -J pll_1node
#BSUB -o pll_1node.o%J
#BSUB -e pll_1node.e%J
#BSUB -W 2:00
#BSUB -nnodes 1

DIR=$HOME/repos/sched
EXE=$DIR/build-vortex/main

source $DIR/load-env.sh

date
jsrun --smpiargs="-gpu" -n 4 -g 1 -c 1 -r 4 -l gpu-gpu,gpu-cpu -b rs \
$EXE \
$HOME/heidi-thornquist-mats/pll.mtx
