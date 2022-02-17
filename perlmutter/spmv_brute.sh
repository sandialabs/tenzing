#!/bin/bash
#SBATCH -A m3953_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 16
#SBATCH --gpus-per-task=1
#SBATCH -e spmv_brute.e%j
#SBATCH -o spmv_brute.o%j
#SBATCH --signal=SIGABRT@10

# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.

DIR=/global/homes/p/pearson/repos/sched
EXE=$DIR/build-perlmutter/src_spmv/spmv-brute

source $DIR/load-env.sh

export SLURM_CPU_BIND="cores"

date

srun -G 4 -n 4 $EXE \
| tee $DIR/perlmutter/spmv_brute.csv
