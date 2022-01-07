#!/bin/bash
#SBATCH -A m3953_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 3:00:00
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 16
#SBATCH --gpus-per-task=1
#SBATCH -e src_spmv_min_time.e%j
#SBATCH -o src_spmv_min_time.o%j
#SBATCH --signal=SIGABRT@10

DIR=/global/homes/p/pearson/repos/sched
EXE=$DIR/build-perlmutter/src_spmv/platform-mcts-min-time

source $DIR/load-env.sh

export SLURM_CPU_BIND="cores"

date

export M=150000

srun -G 4 -n 4 $EXE \
-i 0 \
-b 100 \
-m $M \
| tee $DIR/perlmutter/perl_1n_4r_${M}_src_spmv_min_time.csv

date
