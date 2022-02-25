#include <doctest/doctest.hpp>

#include <iostream>

#include "tenzing/graph.hpp"
#include "tenzing/operation.hpp"
#include "tenzing/platform.hpp"
#include "tenzing/state.hpp"

TEST_CASE("graph with no-op") {

  Platform plat(MPI_COMM_WORLD);

  auto op1 = std::make_shared<NoOp>("op1");

  Graph<OpBase> graph;
  graph.start_then(op1);
  graph.then_finish(op1);

  SDP::State initialState(graph);
  CHECK(initialState.sequence().size() == 1);

  std::vector<std::shared_ptr<Decision>> decisions = initialState.get_decisions(plat);

  // look for Execute op1 exactly once in the decisions
  {
    int count = 0;
    for (const auto &dp : decisions) {
      std::cerr << dp->desc() << "\n";
      if (auto eo = std::dynamic_pointer_cast<ExecuteOp>(dp)) {
        count += (*eo == ExecuteOp(op1));
      }
    }
    CHECK(count == 1);
  }

  for (const auto &dp : decisions) {
    std::cerr << dp->desc() << "\n";

    SDP::State nextState = initialState.apply(*dp);
    CHECK(nextState.sequence().size() == 2);
  }
}