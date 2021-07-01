#pragma once

#include "operation.hpp"

#include <set>
#include <vector>

class Schedule
{
public:
    std::set<Node *> remaining; // possible next operations. Not all have been converted to a CPu node, necessarily
    std::vector<CpuNode *> order; // the order the operations will run in

    void run()
    {
        for (CpuNode *op : order)
        {
            op->run();
        }
    }
};

std::vector<Schedule> make_schedules(Node *start);
