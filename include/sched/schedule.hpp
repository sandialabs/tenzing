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
    std::set<std::shared_ptr<BoundOp>, OpBase::compare_lt> remaining; // possible next operations. Not all have been converted to a CPu node, necessarily
    std::vector<std::shared_ptr<BoundOp>> order; // the order the operations will run in

    void run(Platform &plat)
    {
        for (std::shared_ptr<BoundOp> op : order)
        {
            op->run(plat);
        }
    }

    /* may end up in a case where the CPU is synchronized with the same stream
       twice without any intervening GPU operations.
       Remove all such syncs

        return the number removed
    */
    static int remove_redundant_syncs(std::vector<std::shared_ptr<BoundOp>> &order);
    int remove_redundant_syncs();


    static bool predicate(const Schedule &a, const Schedule &b);

    /* true if the order of nodes in a < b
    */
    static bool by_op_typeid(const Schedule &a, const Schedule &b);

};

std::vector<Schedule> make_schedules(Graph<CpuOp> &g);

// create n random schedules 
// outputs may be repeated
std::vector<Schedule> make_schedules_random(Graph<CpuOp> &g, size_t n);
