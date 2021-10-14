#!/bin/bash
#BSUB -J tio_4node
#BSUB -o tio_4node.o%J
#BSUB -e tio_4node.e%J
#BSUB -W 4:00
#BSUB -nnodes 4

DIR=$HOME/repos/sched
EXE=$DIR/build-vortex/main

source $DIR/load-env.sh

date
jsrun --smpiargs="-gpu" -n 16 -g 1 -c 1 -r 4 -l gpu-gpu,gpu-cpu -b rs \
$EXE \
$HOME/heidi-thornquist-mats/tio.mtx

