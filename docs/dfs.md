# tenzing-dfs

## Examples

* [examples/spmv](examples/spmv.cu): Exploring SpMV implementation space.

## Documentation

1. Use [tenzing-core](github.com/sandialabs/tenzing-core) to specify a `Graph<OpBase>`
2. Construct a `tenzing::EmpiricalBenchmarker` (or other Benchmarker)
3. Construct a `tenzing::Platform`
3. Provide that graph, platform, and benchmarker to `tenzing::dfs::explore(...)`.

* `tenzing::dfs::explore<Benchmarker>(graph, platform, benchmarker)`

### `tenzing::dfs::explore<Benchmarker>(Graph graph, Platform platform, Benchmarker benchmarker, Opts opts)`

Uses depth-first search of the state space, benchmarking each complete implementation to produce a `SimResult` and returning a `Result` consisting of all explored `SimResults`

### `tenzing::dfs::SimResult`

A `tenzing::Sequence` and the corresponding `Benchmark::Result`

### `tenzing::dfs::Result`

A vector of `SimResult`

### `tenzing::dfs::Opts`

* `tenzing::BenchmarkLLOpts benchOpts`: control the behavior of the Benchmarker
* `int64_t maxSeqs`: Generate no more than this many sequences during DFS. These sequences may not be unique.


## Copyright and License

Please see [NOTICE.md](NOTICE.md) for copyright and license information.