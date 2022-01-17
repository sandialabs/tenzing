#!/bin/bash
#BSUB -J 12nodes
#BSUB -o 12nodes.o%J
#BSUB -e 12nodes.e%J
#BSUB -W 2:00
#BSUB -nnodes 12

# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.

DIR=/ascldap/users/cwpears/repos/sched
OUT=$DIR/12node.txt
EXE=$DIR/build-vortex/main

source $DIR/load-env.sh

date
jsrun -d plane:2 --smpiargs="-gpu" -n 48 -g 1 -c 1 -r 4 -l gpu-gpu,gpu-cpu -b rs $EXE

