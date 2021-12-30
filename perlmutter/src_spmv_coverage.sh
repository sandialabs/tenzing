#!/bin/bash
#SBATCH -A m3953_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 16
#SBATCH --gpus-per-task=1
#SBATCH -e src_spmv_coverage.e%j
#SBATCH -o src_spmv_coverage.o%j
#SBATCH --signal=SIGABRT@10

DIR=/global/homes/p/pearson/repos/sched
EXE=$DIR/build-perlmutter/src_spmv/platform-mcts-coverage

source $DIR/load-env.sh

export SLURM_CPU_BIND="cores"

date

srun -G 4 $EXE -i 0 \
| tee $DIR/perlmutter/src_spmv_coverage.csv
