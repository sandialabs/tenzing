/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include "operation.hpp"
#include "graph.hpp"
#include "sequence.hpp"

#include <set>
#include <vector>

class Schedule
{
public:
    std::set<std::shared_ptr<BoundOp>, OpBase::compare_lt> remaining; // possible next operations. Not all have been converted to a CPu node, necessarily
    Sequence<BoundOp> order; // the order the operations will run in

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
    static int remove_redundant_syncs(Sequence<BoundOp> &order);
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
