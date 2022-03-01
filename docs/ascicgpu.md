## ascicgpu030

**CUDA-aware OpenMPI**
```
./configure --with-cuda=/projects/sierra/linux_rh7/SDK/compilers/nvidia/cuda_10.1.243 --with-hwloc=internal CC=gcc CXX=g++
```

**build**

You need a CUDA-aware MPI, which the system OpenMPI is not

To use a different MPI, you might have to unload the system MPI.

Build like this:

```
cmake .. -DCMAKE_CXX_COMPILER=`which g++` -DCMAKE_PREFIX_PATH=$HOME/software/openmpi-4.1.1-cuda10.1-gcc7.2/
```

Then run like


`$HOME/software/openmpi-4.1.1-cuda10.1-gcc7.2/bin/mpirun -n 2 ./main ...`