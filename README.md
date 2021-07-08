## ascicgpu

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

## vortex

CUDA 10.1 & cmake 3.18.0 together cause some problem with recognizing the `-pthread` flag.

```
module --force purge
module load cmake/3.18.0
module load cuda/10.2.89
module load gcc/7.3.1
module load spectrum-mpi/rolling-release
```

**Build**

```
cmake .. -DCMAKE_CUDA_ARCHITECTURES=70
make
```

**run**

```
bsub -W 5:00 -nnodes 1 --shared-launch -Is bash
```

* `-n`: number of resource sets
* `-g`: gpus per rs
* `-c`: cpus per rs
* `-r`: rs per host
* `-l`: latency priority
  * `gpu-gpu,gpu-cpu`: create resource sets by first choose GPUs that minimize latency, then to tiebreak minimize GPU-CPU latency
* `-b`: binding
  * `rs` bind all tasks to the resource set as a whole, not within the resource set.
* vortex has SMT-4, and we'll just use a single CPU (which has 4 threads)
```
jsrun --smpiargs="-gpu" -n 2 -g 1 -c 1 -r 2 -l gpu-gpu,gpu-cpu -b rs ./main
```