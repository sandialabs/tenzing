#!/bin/bash
#SBATCH -A m3918_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 4:00:00
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 16
#SBATCH --gpus-per-task=1
#SBATCH -e spmv_whole_tree.e%j
#SBATCH -o spmv_whole_tree.o%j

DIR=/global/homes/p/pearson/repos/sched
EXE=$DIR/build-perlmutter/src_spmv/platform-mcts-random

source $DIR/load-env.sh

export SLURM_CPU_BIND="cores"

srun -G 4 $EXE -i 0 \
| tee $DIR/perlmutter/spmv_whole_tree.csv

