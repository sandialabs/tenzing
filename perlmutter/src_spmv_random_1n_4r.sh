#!/bin/bash
#SBATCH -A m3953_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 16
#SBATCH --gpus-per-task=1
#SBATCH -e src_spmv_random.e%j
#SBATCH -o src_spmv_random.o%j
#SBATCH --signal=SIGABRT@10

# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.

DIR=/global/homes/p/pearson/repos/sched
EXE=$DIR/build-perlmutter/src_spmv/platform-mcts-random

source $DIR/load-env.sh

export SLURM_CPU_BIND="cores"

hostname
date

export M=150000

srun -G 4 -n 4 $EXE \
-i 0 \
-b 100 \
-m $M \
| tee $DIR/perlmutter/src_spmv_random_1n_4r_${M}.csv

date
