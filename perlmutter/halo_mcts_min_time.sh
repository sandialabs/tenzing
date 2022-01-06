#!/bin/bash
#SBATCH -A m3953_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 16
#SBATCH --gpus-per-task=1
#SBATCH -e halo_mcts_min_time.e%j
#SBATCH -o halo_mcts_min_time.o%j
#SBATCH --signal=SIGABRT@10

DIR=/global/homes/p/pearson/repos/sched
EXE=$DIR/build-perlmutter/src_mcts_halo/halo-mcts-min-time

source $DIR/load-env.sh

export SLURM_CPU_BIND="cores"

date

srun -G 4 -n 4 $EXE \
| tee $DIR/perlmutter/halo_mcts_min_time_1n_4r.csv
