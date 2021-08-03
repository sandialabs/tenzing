#! /bin/bash

host=`hostname`

if [[ "$host" =~ .*ascicgpu.* ]]; then
    echo "$host" matched ascicgpu
    
    echo "export CUDAARCHS=70"
    export CUDAARCHS="70" # for cmake 3.20+

    echo "module purge"
    module purge

    echo "module load sierra-devel/nvidia"
    module load sierra-devel/nvidia

    echo "module load cde/v2/cmake/3.19.2"
    module load cde/v2/cmake/3.19.2

    which cmake
    which gcc
    which nvcc
    which mpirun
elif [[ "$host" =~ .*vortex.* ]]; then
# CUDA 10.1 & cmake 3.18.0 together cause some problem with recognizing the `-pthread` flag.

    echo "$host" matched vortex
    
    echo "export CUDAARCHS=70"
    export CUDAARCHS="70" # for cmake 3.20+

    echo module --force purge
    module --force purge

    echo module load cmake/3.18.0
    module load cmake/3.18.0
    echo module load cuda/10.2.89
    module load cuda/10.2.89
    echo module load gcc/7.3.1
    module load gcc/7.3.1
    echo module load spectrum-mpi/rolling-release
    module load spectrum-mpi/rolling-release

    which cmake
    which gcc
    which nvcc
    which mpirun
fi
