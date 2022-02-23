/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "sched/numeric.hpp"
#include "sched/operation.hpp"
#include "sched/ops_mpi.hpp"
#include "sched/schedule.hpp"
#include "sched/graph.hpp"
#include "sched/mcts.hpp"
#include "sched/mcts_strategy_min_time.hpp"

#include <mpi.h>

#include <vector>
#include <memory>

#include <thread> //sleep
#include <chrono>

int main(int argc, char **argv) {

    sched::init();

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    STDERR("create buffers...");
    std::vector<char> sbuf(1000ul * 1000ul, 0);
    auto rbuf = sbuf;

    STDERR("create nodes...");
    std::shared_ptr<Start> start = std::make_shared<Start>();
    std::shared_ptr<End> end = std::make_shared<End>();

    auto owa1 = std::make_shared<OwningWaitall>(2, "owa1");
    auto owa2 = std::make_shared<OwningWaitall>(2, "owa2");

    int dest = (rank + 1) % size;
    int source = (rank - 1);
    if (source < 0) source += size;

    auto is1 = std::make_shared<Isend>(Isend::Args{
        .buf=sbuf.data(),
        .count=int(sbuf.size()),
        .datatype=MPI_CHAR,
        .dest=dest,
        .tag=0,
        .comm=MPI_COMM_WORLD,
        .request=&owa1->requests()[0]
    }, "is1");
    auto ir1 = std::make_shared<Irecv>(Irecv::Args{
        .buf=rbuf.data(),
        .count=int(rbuf.size()),
        .datatype=MPI_CHAR,
        .source=source,
        .tag=0,
        .comm=MPI_COMM_WORLD,
        .request=&owa1->requests()[1]
    }, "ir1");
    auto is2 = std::make_shared<Isend>(Isend::Args{
        .buf=sbuf.data(),
        .count=int(sbuf.size()),
        .datatype=MPI_CHAR,
        .dest=dest,
        .tag=0,
        .comm=MPI_COMM_WORLD,
        .request=&owa2->requests()[0]
    }, "is2");
    auto ir2 = std::make_shared<Irecv>(Irecv::Args{
        .buf=rbuf.data(),
        .count=int(rbuf.size()),
        .datatype=MPI_CHAR,
        .source=source,
        .tag=0,
        .comm=MPI_COMM_WORLD,
        .request=&owa2->requests()[1]
    }, "ir2");

    STDERR("create graph...");
    Graph<CpuNode> orig(start);
    orig.then(start, is1);
    orig.then(start, ir1);
    orig.then(is1, owa1);
    orig.then(ir1, owa1);
    orig.then(owa1, is2);
    orig.then(owa1, ir2);
    orig.then(is2, owa2);
    orig.then(ir2, owa2);
    orig.then(owa2, end);

    STDERR("mcts...");
    mcts::Opts opts;
    opts.dumpTreeEvery = 10;
    opts.dumpTreePrefix = "pingpong";
    opts.nIters = 1000;
    opts.benchOpts.nIters = 100;
    EmpiricalBenchmarker benchmarker;
    mcts::mcts<mcts::FastMin>(orig, benchmarker, MPI_COMM_WORLD, opts);

    MPI_Finalize();
}