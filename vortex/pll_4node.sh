#!/bin/bash
#BSUB -J pll_4node
#BSUB -o pll_4node.o%J
#BSUB -e pll_4node.e%J
#BSUB -W 2:00
#BSUB -nnodes 4

DIR=$HOME/repos/sched
EXE=$DIR/build-vortex/main

source $DIR/load-env.sh

date
jsrun --smpiargs="-gpu" -n 16 -g 1 -c 1 -r 4 -l gpu-gpu,gpu-cpu -b rs \
$EXE \
$HOME/heidi-thornquist-mats/pll.mtx

