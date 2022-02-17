#!/bin/bash
#BSUB -J 16node
#BSUB -o 16node.o%J
#BSUB -e 16node.e%J
#BSUB -W 2:00
#BSUB -nnodes 16
##BSUB -R select[hname!='vortex10']

# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.

DIR=/ascldap/users/cwpears/repos/sched
OUT=$DIR/16node.txt
EXE=$DIR/build-vortex/main

source $DIR/load-env.sh

date
jsrun --smpiargs="-gpu" -n 64 -g 1 -c 1 -r 4 -l gpu-gpu,gpu-cpu -b rs $EXE

