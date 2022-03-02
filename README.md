# tenzing-core

The core library of the Tenzing project.
tenzing-core provides facilities for interacting with CUDA + MPI programs as sequential decision problems.
This facilitates optimizing CUDA + MPI programs using sequential decision strategies.

Two solvers are available
* [tenzing-mcts](https://github.com/sandialabs/tenzing-mcts):  Uses Monte-Carlo tree search
* [tenzing-dfs](https://github.com/sandialabs/tenzing-dfs): Uses depth-first search

## Build

On a supported platform:
```bash
source load-env.sh
```

In any case:
```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=70
make
```

| Option | Default | Meaning|
|-|-|-|
| `-DTENZING_ENABLE_TESTS` | `ON` | Compile tests |
| `-DTENZING_BUILD_PYTHON_BINDINGS` | `OFF` | Compile Python bindings (requires Python development headers/libraries)

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

tenzing-core has been tested on the following platforms:
* NERSC perlmutter: g++ 10.3 / nvcc 11.4 / Cray MPICH 8.1.13
* Sandia vortex (similar to ORNL Lassen and OLCF Summit): g++ 7.5.0 / nvcc 10.1 / IBM Spectrum MPI
* Sandia ascicgpu

## Documentation

* Visit the API documentation in [docs/api.md](docs/api.md)
* `ascicgpu` system documentation in [docs/ascicgpu.md](docs/ascicgpu.md)
* `vortex` system documentation in [docs/vortex.md](docs/vortex.md)
* `perlmutter` ssytem documentation in [docs/perlmutter.md](docs/perlmutter.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

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

## Copyright and License

Please see [NOTICE.md](https://github.com/sandialabs/tenzing-core/blob/master/NOTICE.md) for copyright and license information.