# API

The general flow is

1. Call `tenzing::init(argc, argv)`
2. Call `tenzing::reproduce::dump_with_cli(argc, argv)` to print version information in your program outputs.
3. Define the MPI and GPU operations of your program using `SDP::OpBase` and it's children
4. Construct an `SDP::Graph<OpBase>` to model the dependences between your program's operations.
5. Construct an initial `SDP::State` from the graph.
6. Construct a model of the execution platform with `SDP::Platform`
7. Use `State::get_decisions(Platform)` to generate list of possible next operations or resource constraints to apply in the program.
8. Use `State::apply(Decision)` to get the state that results from a particular decision.

## Operations

- `SDP::BaseOp`: A vertex in a DAG representing a program.
  - `SDP::CpuOp`: A `BaseOp` that represents an operation that the control thread can actually execute. May represent the start of an asynchronous operation (`MPI_Isend`, CUDA kernel launch).
    - `SDP::BoundGpuOp`: A `CpuOp` made of a `GpuOp` (below) with an associated stream
  - `SDP::GpuOp`: An operation that needs to be bound to a stream before it can be executed.
  - `SDP::CompoundOp`:
  - `SDP::ChoiceOp`:
- `SDP:Graph`: A graph, where vertices are usually `BaseOp` and edge *u* -> *v* means *u* must happen before *v*.

## `SDP::Graph<T>`

Typically `Graph<OpBase>`.
A graph representing the dependences between operations.
Each vertex is an `std::shared_ptr<T>`, and each edge *u* -> *v* means *u* must happen before *v*.

* `Graph::start_then(const std::shared_ptr<OpBase> op)`:
* `Graph::then_finish(const std::shared_ptr<OpBase> op)`:
* `Graph::then(const std::shared_ptr<OpBase> op)`:
* `Graph::clone()`:
* `Graph::clone_but_replace(...)`:
* `Graph::clone_but_expand(...)`:

## `SDP::State`

A `Sequence<BoundOp>` of a partial program order paired with a `Graph<BaseOp>` representing the constrained program at this point.

* `State::State(const Graph<OpBase> &graph)`: construct an initial state from a graph
* `State::sequence()`: access the sequence in this state
* `State::graph()`: access the graph in this state

## `SDP::Sequence`

Typically `Sequence<BoundOp>`.
An executable sequence of operations.


## `SDP::Decision`

- `SDP::Decision`: `State` knows how to use a `Decision` to produce a new `State`s. May represent binding a particular `GpuOp` to a stream, or executing a ready `CpuOp`, or expanding a `CompoundOp`.

A consequence of this implementation is that not all `Decisions` actually represent the execution of a program operation.
They may instead constrain the program in some way (a resource binding, choosing among multiple implementation options).
The next `State` resulting from the current `State` and a `Decision` would reflect such a change in a revised `Graph` in that `State`.

## Inernal Components

### `EventSynchronizer`

Knows how to tell if two operations in a sequence are synchronized, and if not, how to synchronize them.
For example, consider BoundGpuOp *a* in stream 1 and *b* in stream 2.

* `EventSynchronizer::is_synced()`: TODO

### Benchmarkers

A benchmarker knows how to turn a Sequence into a performance measurement.

#### `SDP::EmpiricalBenchmarker`
runs the schedule on the machine and reports the result
#### `SDP::CsvBenchmarker`
looks the schedule up in a CSV file of times and uses that as the result