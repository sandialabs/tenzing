#! /bin/bash

host=`hostname`

if [[ "$host" =~ .*ascicgpu.* ]]; then
    echo "$host" matched ascicgpu

    echo "module purge"
    module purge

    echo "module load sierra-devel/nvidia"
    module load sierra-devel/nvidia

    echo "module load cde/v2/cmake/3.19.2"
    module load cde/v2/cmake/3.19.2

    which gcc
    which nvcc
    which mpirun
fi