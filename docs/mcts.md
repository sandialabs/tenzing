## Examples

* [examples/spmv-coverage](examples/spmv_coverage.cu): MCTS using "coverage" strategy
* [examples/spmv-random](examples/spmv_random.cu): ... "random" ...
* [examples/spmv-min-time](examples/spmv_min_time.cu): ... "MinTime" ...

## Documentation

1. Use [tenzing-core](github.com/sandialabs/tenzing-core) to specify a `Graph<OpBase>`
2. Construct a `tenzing::EmpiricalBenchmarker` (or other Benchmarker)
3. Construct a `tenzing::Platform`
4. Provide that graph, platform, and benchmarker to `tenzing::mcts::explore(...)`.

### Strategies

Strategies affect how the `exploit` part of the explore/exploit score is calculated for MCTS.