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

## Benchmarkers
A benchmarker knows how to turn a Schedule into a performance measurement.
* `EmpiricalBenchmarker` runs the schedule on the machine and reports the result
* `CsvBenchmarker` looks the schedule up in a CSV file of times and uses that as the result

## Binaries

| source | Description |
|-|-|
| `src_spmv/spmv_brute
| `src_spmv/platform_mcts.cu` | SpMV Uses MCTS to explore stream assignment and operation ordering |
| `src_spmv/mcts_csv_coverage.cu` | SpMV. Uses MCTS-coverage and loads times from CSV file |
| `src_spmv/mcts_csv_coverage.cu` | SpMV. Uses MCTS-coverage and loads times from CSV file |

## Tests

Tests are split into two locations:
* unit tests may be defined in source files
* tests with a more "itegration" flavor are in `test/`

To run tests, you can do
* `make test`
* `ctest`
* `tenzing-all`
  * `-ltc`: list tests cases
  * `-tc="a,b"`: only run test cases named `a` and `b`

This creates some CMake complexity, as the test functions present in static libraries will not be linked into the resulting test binary.
Therefore, we use a CMake object library to generate the test binary, and then generate a static library from the object library.
object library properties do not get propagated properly / at all, so we have to redefine what needs to be linked and included, etc

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

| project | name |
|-|-|
|m3953| FASTMath |
|m3918| KokkosKernels |

**interactive run**

* `srun -G`: total number of GPUs for the job
* `srun -n`: number of MPI tasts for the job

```
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m3953_g
srun -G 4 -n 4 src_spmv/platform-mcts-random
```

**run**

* `sbatch -n`: number of tasks
* `sbatch -c`: number of CPUs per task
* `sbatch --ntasks-per-node`: tasks per node
* `sbatch -e/-o `: stderr/stdout file. use `%j` to insert jobid

* `sqs`: monitor job queue
* `scontrol show job <jobid> | grep -oP  'NodeList=nid(\[.+\]|.+)'` get a list of nodes for a job
  * `ssh <node>` ssh into one of those nodes 

* `sbatch perlmutter/<script.sh>`: submit a predefined script

**issues**

CUDA-aware MPI may not interact well with `cuda-memcheck`

```
========= Program hit CUDA_ERROR_INVALID_VALUE (error 1) due to "invalid argument" on CUDA API call to cuPointerGetAttribute.
=========     Saved host backtrace up to driver entry point at error
=========     Host Frame:/usr/local/cuda-11.5/compat/libcuda.so.1 [0x232f1c]
=========     Host Frame:/opt/cray/pe/mpich/8.1.13/gtl/lib/libmpi_gtl_cuda.so.0 [0x9ed0]
=========     Host Frame:/opt/cray/pe/mpich/8.1.13/gtl/lib/libmpi_gtl_cuda.so.0 (mpix_gtl_pointer_type + 0x9) [0x2e99]
=========     Host Frame:/opt/cray/pe/mpich/8.1.13/ofi/nvidia/20.7/lib/libmpi_nvidia.so.12 [0x60f26a]
=========     Host Frame:/opt/cray/pe/mpich/8.1.13/ofi/nvidia/20.7/lib/libmpi_nvidia.so.12 [0x1a71763]
=========     Host Frame:/opt/cray/pe/mpich/8.1.13/ofi/nvidia/20.7/lib/libmpi_nvidia.so.12 [0x1b13a9f]
=========     Host Frame:/opt/cray/pe/mpich/8.1.13/ofi/nvidia/20.7/lib/libmpi_nvidia.so.12 [0x232bd0]
=========     Host Frame:/opt/cray/pe/mpich/8.1.13/ofi/nvidia/20.7/lib/libmpi_nvidia.so.12 (PMPI_Barrier + 0x16f) [0x23310f]
=========     Host Frame:src_spmv/spmv-brute (main + 0x3d1) [0x27e51]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xea) [0x2434a]
=========     Host Frame:src_spmv/spmv-brute (_start + 0x2a) [0x2875a]
```


## Design

Core library for representing the design CUDA + MPI program as a Sequental Decision Problem (SDP)

- `SDP::BaseOp`: A vertex in a DAG representing a program.
  - `SDP::CpuOp`: A `BaseOp` that represents an operation that the control thread can actually execute. May represent the start of an asynchronous operation (`MPI_Isend`, CUDA kernel launch).
    - `SDP::BoundGpuOp`: A `CpuOp` made of a `GpuOp` (below) with an associated stream
  - `SDP::GpuOp`: An operation that needs to be bound to a stream before it can be executed.
  - `SDP::CompoundOp`:
  - `SDP::ChoiceOp`:
- `SDP:Graph`: A graph, where vertices are usually `BaseOp` and edge *u* -> *v* means *u* must happen before *v*.
- `SDP::Sequence`: 
- `SDP::State`: A `Sequence<BoundOp` of a partial program order paired with a `Graph<BaseOp>` representing the entire program at this point.
- `SDP::Decision`: `State` knows how to use a `Decision` to produce a new `State`s. May represent binding a particular `GpuOp` to a stream, or executing a ready `CpuOp`, or expanding a `CompoundOp`.

A consequence of this implementation is that not all `Decisions` actually represent the execution of a program operation.
They may instead constrain the program in some way (a resource binding, choosing among multiple implementation options).
The next `State` resulting from the current `State` and a `Decision` would reflect such a change in a revised `Graph` in that `State`.


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

## Copyright and License

Please see [NOTICE.md](https://github.com/sandialabs/tenzing-core/blob/master/NOTICE.md) for copyright and license information.