#pragma once

#include "operation.hpp"
#include "graph.hpp"

#include <set>
#include <vector>

#include <mpi.h>

struct BenchOpts {
    size_t nIters;

    BenchOpts() : nIters(1000) {}
};

class Schedule
{
public:
    std::set<std::shared_ptr<CpuNode>, Node::compare_lt> remaining; // possible next operations. Not all have been converted to a CPu node, necessarily
    std::vector<std::shared_ptr<CpuNode>> order; // the order the operations will run in

    void run()
    {
        for (std::shared_ptr<CpuNode> op : order)
        {
            op->run();
        }
    }

    /* may end up in a case where the CPU is synchronized with the same stream
       twice without any intervening GPU operations.
       Remove all such syncs

        return the number removed
    */
    static int remove_redundant_syncs(std::vector<std::shared_ptr<CpuNode>> &order);
    int remove_redundant_syncs();


    static bool predicate(const Schedule &a, const Schedule &b);

    /* true if the order of nodes in a < b
    */
    static bool by_node_typeid(const Schedule &a, const Schedule &b);

    struct BenchResult {
        double pct01;
        double pct10;
        double pct50;
        double pct90;
        double pct99;
        double stddev;
    };



    static std::vector<BenchResult> benchmark(std::vector<Schedule> &schedules, MPI_Comm comm, const BenchOpts &opts = BenchOpts()); 
    static BenchResult benchmark(std::vector<std::shared_ptr<CpuNode>> &order, MPI_Comm comm, const BenchOpts &opts = BenchOpts());
};

std::vector<Schedule> make_schedules(Graph<CpuNode> &g);

// create n random schedules 
// outputs may be repeated
std::vector<Schedule> make_schedules_random(Graph<CpuNode> &g, size_t n);


