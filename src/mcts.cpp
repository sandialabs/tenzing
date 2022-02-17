/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "sched/mcts.hpp"

#include "sched/operation_serdes.hpp"
#include "sched/schedule.hpp"

namespace mcts {


void Result::dump_csv() const {

    const std::string delim("|");

    for (size_t i = 0; i < simResults.size(); ++i) {
        const auto &simres = simResults[i];
        std::cout << i;
        std::cout << delim << simres.benchResult.pct01;
        std::cout << delim << simres.benchResult.pct10;
        std::cout << delim << simres.benchResult.pct50;
        std::cout << delim << simres.benchResult.pct90;
        std::cout << delim << simres.benchResult.pct99;
        std::cout << delim << simres.benchResult.stddev;
        for (const auto &op : simres.path) {
            std::cout << "|" << op->json();
        }
        std::cout << "\n"; 
    }
}


} // namespace mcts