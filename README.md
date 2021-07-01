# ascicgpu

```
module purge
module load sierra-devel/nvidia
module load cde/v2/cmake/3.19.2
```

To use a different MPI, you might have to unload the system MPI.

Build like this:

```
cmake .. -DCMAKE_CXX_COMPILER=`which g++` -DCMAKE_PREFIX_PATH=$HOME/software/openmpi-4.1.1-cuda10.1-gcc7.2/
```