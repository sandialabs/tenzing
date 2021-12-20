#! /bin/bash

host=`hostname`

if [[ "$host" =~ .*ascicgpu.* ]]; then
    echo "$host" matched ascicgpu
    
    echo "export CUDAARCHS=70"
    export CUDAARCHS="70" # for cmake 3.20+

    echo "module purge"
    module purge

    mkdir -p /tmp/$USER
    export TMPDIR=/tmp/$USER

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
    export NUMA_ROOT_DIR=$HOME/software/numactl2.0.14-gcc7.3.1

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
elif [[ "$host" =~ .*weaver.* ]]; then
# CUDA 10.1 & cmake 3.18.0 together cause some problem with recognizing the `-pthread` flag.

    echo "$host" matched weaver
    
    echo "export CUDAARCHS=70"
    export CUDAARCHS="70" # for cmake 3.20+

    echo module --force purge
    module --force purge

    echo module load cmake/3.19.3
    module load cmake/3.19.3
    echo module load cuda/10.2.2
    module load cuda/10.2.2
    echo module load gcc/7.2.0
    module load gcc/7.2.0
    echo module load openmpi/4.0.5
    module load openmpi/4.0.5

    which cmake
    which gcc
    which nvcc
    which mpirun
elif [[ "$NERSC_HOST" =~ .*perlmutter.* ]]; then
# CUDA 10.1 & cmake 3.18.0 together cause some problem with recognizing the `-pthread` flag.

    echo "$NERSC_HOST" matched perlmutter
    
    echo "export CUDAARCHS=70"
    export CUDAARCHS="70" # for cmake 3.20+
    echo "export MPICH_GPU_SUPPORT_ENABLED=1"
    export MPICH_GPU_SUPPORT_ENABLED=1


    echo module load cmake/3.22.0
    module load cmake/3.22.0
    echo module load nvidia/21.9
    module load nvidia/21.9


    which cmake
    which gcc
    which nvcc
    which mpirun
fi
