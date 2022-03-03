/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "tenzing/dfs/dfs.hpp"

namespace tenzing {
namespace dfs {

std::vector<Sequence<BoundOp>> get_all_sequences(const Graph<OpBase> &g, Platform &plat,
                                                 int64_t maxSeqs) {
  std::vector<SDP::State> worklist;
  std::vector<Sequence<BoundOp>> ret;

  auto boundStart = std::dynamic_pointer_cast<BoundOp>(g.start());
  if (!boundStart) {
    THROW_RUNTIME("");
  }

  SDP::State initial(g, {boundStart});
  worklist.push_back(initial);

  while (!worklist.empty()) {

    STDERR("get_all_sequences: worklist " << worklist.size() << " complete " << ret.size());

    if (ret.size() >= maxSeqs) {
      break;
    }

    // DFS
    SDP::State curr = worklist.back();
    worklist.pop_back();

    // get the frontier from the current state
    std::vector<SDP::State> frontier = curr.frontier(plat, true);

    // especially at the beginning of the search, some elements in the frontier may be equivalent
    // no need to search them all
    {
      size_t numEq = 0;
      for (auto fi = frontier.begin(); fi < frontier.end(); ++fi) {
        for (auto fj = fi + 1; fj < frontier.end(); ++fj) {
          if (get_equivalence(*fi, *fj)) {
            frontier.erase(fj);
            ++numEq;
            --fj;
          }
        }
      }
      STDERR(numEq << " equivalents removed");
    }

#if 0
    {
      std::string s;
      for (const auto &op : frontier) {
        s += op->desc();
        s += ", ";
      }
      STDERR("get_all_sequences: graph frontier is: " << s);
    }
#endif

    if (frontier.empty()) { // this state is complete
      ret.push_back(curr.sequence());
    } else {

      for (const SDP::State &state : frontier) {
        worklist.push_back(state);
      }
    }
  }

  return ret;
}

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
    for (const auto &op : simres.seq) {
      std::cout << "|" << op->json();
    }
    std::cout << "\n";
  }
}

} // namespace dfs
} // namespace tenzing