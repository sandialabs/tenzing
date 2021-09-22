#include "sched/numeric.hpp"
#include "sched/operation.hpp"
#include "sched/schedule.hpp"
#include "sched/graph.hpp"
#include "sched/mcts.hpp"

#include <mpi.h>

#include <vector>
#include <memory>
#include <algorithm>

int main(void) {
    std::cerr << "create nodes...\n";
    std::shared_ptr<Start> start = std::make_shared<Start>();
    std::shared_ptr<NoOp> noop = std::make_shared<NoOp>("NoOp");
    std::shared_ptr<End> end = std::make_shared<End>();

    std::cerr << "create graph...\n";
    Graph<CpuNode> orig(start);
    orig.then(start, noop);
    orig.then(noop, end);

    std::cerr << "mcts...\n";
    mcts::mcts(orig);
}