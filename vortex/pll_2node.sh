#!/bin/bash
#BSUB -J pll_2node
#BSUB -o pll_2node.o%J
#BSUB -e pll_2node.e%J
#BSUB -W 2:00
#BSUB -nnodes 2

DIR=$HOME/repos/sched
EXE=$DIR/build-vortex/main

source $DIR/load-env.sh

date
jsrun --smpiargs="-gpu" -n 8 -g 1 -c 1 -r 4 -l gpu-gpu,gpu-cpu -b rs \
$EXE \
$HOME/heidi-thornquist-mats/pll.mtx

