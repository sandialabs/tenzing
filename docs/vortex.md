# Vortex

## Build

```
export NUMA_ROOT_DIR=...
cmake ..
```

## Run

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