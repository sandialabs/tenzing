#pragma once

#include "operation.hpp"
#include "graph.hpp"

#include <set>
#include <vector>

class Schedule
{
public:
    std::set<std::shared_ptr<CpuNode>> remaining; // possible next operations. Not all have been converted to a CPu node, necessarily
    std::vector<std::shared_ptr<CpuNode>> order; // the order the operations will run in

    void run()
    {
        for (std::shared_ptr<CpuNode> op : order)
        {
            op->run();
        }
    }

    static bool predicate(const Schedule &a, const Schedule &b);

    /* true if the order of nodes in a < b
    */
    static bool by_node_typeid(const Schedule &a, const Schedule &b);
};

std::vector<Schedule> make_schedules(Graph<CpuNode> &g);
