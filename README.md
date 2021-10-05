**Build**

```
source ../load-env.sh
cmake .. -DCMAKE_CUDA_ARCHITECTURES=70
make
```

* `src_halo`: halo exchange, brute force stream assignments, N random schedules
* `src_mcts_halo` halo exchange, given stream assignment, MCTS schedules
* `src_mcts_spmv` spmv, given stream assignment, MCTS schedules
* `src_spmv`: halo exchange, brute force stream assignmts, brute force schedules


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

**run**

## vortex




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

## Organization

* `src`: library source files
* `include`: library include files

* `src_spmv`: sources for the SpMV operation
* `src_halo`: sources for the halo exchange operation

## Extending

Each class of node needs a unique `tag()` for sorting.
Be sure that no newly defined node has a `tag()` function that returns the same value as any other class of node