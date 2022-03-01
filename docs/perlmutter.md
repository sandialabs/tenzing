# Perlmutter

| project | name |
|-|-|
|m3953| FASTMath |
|m3918| KokkosKernels |

## Build

```
source ../load-env.sh
cmake ..
```



## interactive run

* `srun -G`: total number of GPUs for the job
* `srun -n`: number of MPI tasts for the job

```
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m3953_g
srun -G 4 -n 4 src_spmv/platform-mcts-random
```

## run

* `sbatch -n`: number of tasks
* `sbatch -c`: number of CPUs per task
* `sbatch --ntasks-per-node`: tasks per node
* `sbatch -e/-o `: stderr/stdout file. use `%j` to insert jobid

* `sqs`: monitor job queue
* `scontrol show job <jobid> | grep -oP  'NodeList=nid(\[.+\]|.+)'` get a list of nodes for a job
  * `ssh <node>` ssh into one of those nodes 

* `sbatch perlmutter/<script.sh>`: submit a predefined script

## issues

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