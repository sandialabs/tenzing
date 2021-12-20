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

## `sched` Library (`include/sched` and `src`)


## Benchmarkers
A benchmarker knows how to turn a Schedule into a performance measurement.
* `EmpiricalBenchmarker` runs the schedule on the machine and reports the result
* `CsvBenchmarker` looks the schedule up in a CSV file of times and uses that as the result

## `static`
Holds some useful static files.

| File | Description |
|-|-|
| `vortex_4n_1r_spmv.csv` | Brute-force search of a 4-node 1 rank-per-node SpMV on Vortex |

## Binaries

| source | Description |
|-|-|
| `src_spmv/platform_mcts.cu` | SpMV Uses MCTS to explore stream assignment and operation ordering |
| `src_spmv/mcts_csv_coverage.cu` | SpMV. Uses MCTS-coverage and loads times from CSV file |
| `src_spmv/mcts_csv_coverage.cu` | SpMV. Uses MCTS-coverage and loads times from CSV file |

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

```
export NUMA_ROOT_DIR=...
cmake ..
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

## perlmutter

```
source ../load-env.sh
cmake ..
```

**run**

## Design

- `OpBase` common interface for all DAG operations
  - `GpuOp` an Op that runs on the GPU. Can't run until it's bound to a resource
  - `BoundOp` represents an Op that has been attached to a resource. adds run(Platform &)
    - `CpuOp` is a BoundOp that runs on the CPU. Right now Platform does not do anythin with CPUs so it's just a BoundOp.
    - `BoundGpuOp` holds a `GpuOp`

## Organization

* `src`: library source files
* `include`: library include files

* `src_spmv`: sources for the SpMV operation
* `src_halo`: sources for the halo exchange operation

## Extending

Each class of node needs a unique `tag()` for sorting.
Be sure that no newly defined node has a `tag()` function that returns the same value as any other class of node

## To Do:

- [ ] Ser/Des 
  - [x] GpuOp::run takes a stream id, not a cudaStream_t
  - [ ] No way to ser/des events right now (lose track of related synchronization operations)
    - [ ] replace events with an id
    - [ ] each rank
  - [ ] Ser/Des for Ops not in graph (inserted sync, etc)


- [x] BoundOp::run() takes a platform argument
- [ ] Reduce event count
  - [ ] once an ordering is decided, cudaEvent_t are bound to operations
- [ ] Multi-GPU support
  - [ ] Operations referencing values they need
    - [ ] Values attached to a resource (CPU or GPU)
    - [ ] These values would be created once the graph is finalized


## Design Issues

- [ ] enable / disable CUDA / MPI
  - [ ] isolate Ser/Des
  - [ ] isolate platform assignments
- [ ] a `BoundOp` cannot produce the `std::shared_ptr<OpBase>` of it's unbound self, only `OpBase`
  - can't ask an `std::shared_ptr<BoundOp>` for `std::shared_ptr<OpBase>`
  - maybe std::shared_from_this?
- [ ] special status of `Start` and `End` is a bit clumsy.
  - maybe there should be a `StartEnd : BoundOp` that they both are instead of separate classes
    - in the algs they're probably treated the same (always synced, etc)
- [ ] `Platform` is a clumsy abstraction, since it also tracks resources that are only valid for a single order
   - e.g., each order requires a certain number of events, which can be resued for the next order

## Ideas

