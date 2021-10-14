#!/bin/bash
#BSUB -J spmv-stream-brute
#BSUB -o spmv-stream-brute.o%J
#BSUB -e spmv-stream-brute.e%J
#BSUB -W 2:00
#BSUB -nnodes 1

DIR=/ascldap/users/cwpears/repos/sched
OUT=$DIR/vortex/spmv-stream-brute.csv
EXE=$DIR/build-vortex/src_spmv/spmv-stream-brute

source $DIR/load-env.sh

date
jsrun --smpiargs="-gpu" -n 4 -g 1 -c 2 -r 4 -l gpu-gpu,gpu-cpu -b packed:1 $EXE | tee $OUT
date