#include <doctest/doctest.hpp>

#include <iostream>

#include "tenzing/graph.hpp"
#include "tenzing/operation.hpp"
#include "tenzing/platform.hpp"
#include "tenzing/state.hpp"

__global__ void kernel(int *x) {}

class KernelOp : public GpuOp {
public:
  KernelOp(const std::string &name) : name_(name) {}
  virtual void run(cudaStream_t stream) override {
    kernel<<<1,1,0,stream>>>(nullptr);
  }

  // OpBase
  virtual std::string name() const override { return name_; }
  bool operator<(const KernelOp &rhs) const { return name_ < rhs.name_; }
  bool operator==(const KernelOp &rhs) const { return name_ == rhs.name_; }
  CLONE_DEF(KernelOp);
  LT_DEF(KernelOp);
  EQ_DEF(KernelOp);

  std::string name_;
};

template <typename Dec>
int count_occurances(const std::vector<std::shared_ptr<Decision>> &ds, const Dec &d) {
  int count = 0;
  for (const auto &dp : ds) {
    if (auto casted = std::dynamic_pointer_cast<Dec>(dp)) {
      count += (*casted == d);
    }
  }
  return count;
}

TEST_CASE("graph with GpuOps") {
  std::cerr << "TEST_CASE graphwithGpuOps\n";

  std::cerr << "Platform::make_n_streams()...\n";
  Platform plat = Platform::make_n_streams(2, MPI_COMM_WORLD);

  std::cerr << "construct kernel operations...\n";
  auto kernel1 = std::make_shared<KernelOp>("kernel1");
  auto kernel2 = std::make_shared<KernelOp>("kernel2");
  auto kernel3 = std::make_shared<KernelOp>("kernel3");

  std::cerr << "build graph...\n";
  Graph<OpBase> graph;
  graph.start_then(kernel1);
  graph.then(kernel1, kernel2);
  graph.then(kernel1, kernel3);
  graph.then_finish(kernel2);
  graph.then_finish(kernel3);

  std::cerr << "initialState...\n";
  SDP::State initialState(graph);
  CHECK(initialState.sequence().size() == 1);

  std::cerr << "initialState.get_decisions()...\n";
  std::vector<std::shared_ptr<Decision>> decisions = initialState.get_decisions(plat);

  for (const auto &dp : decisions) {
    std::cerr << dp->desc() << "\n";
  }

  // kernel 1 should be assigned to each stream
  CHECK(1 == count_occurances(decisions, AssignOpStream(kernel1, Stream(0))));
  CHECK(1 == count_occurances(decisions, AssignOpStream(kernel1, Stream(1))));

  
  SUBCASE("kernel1 in stream 0") {
    std::cerr << "SUBCASE kernel1 in stream 0\n";

    STDERR("apply kernel 1 in stream 0...");
    SDP::State state = initialState.apply(AssignOpStream(kernel1, Stream(0)));
    CHECK(1 == state.sequence().size());

    // should be equivalent to kernel 1 in stream 1
    {
      STDERR("apply kernel 1 in stream 1...");
      SDP::State tmp = initialState.apply(AssignOpStream(kernel1, Stream(1)));
      STDERR("check sequence equivalence...");
      CHECK(true == bool(get_equivalence(tmp.sequence(), state.sequence())));
      STDERR("check graph equivalence...");
      CHECK(true == bool(get_equivalence(tmp.graph(), state.graph())));
      STDERR("check state equivalence...");
      CHECK(true == bool(get_equivalence(tmp, state)));
    }

    // graph should contain a bound version of kernel 1 with other unbound
    auto gkernel1 = std::make_shared<BoundGpuOp>(kernel1, Stream(0));
    CHECK(state.graph().contains(gkernel1));
    CHECK(state.graph().contains(kernel2));
    CHECK(state.graph().contains(kernel3));
    
    decisions = state.get_decisions(plat);
    for (const auto &dp : decisions) {
      std::cerr << dp->desc() << "\n";
    }

    CHECK(1 == count_occurances(decisions, ExecuteOp(gkernel1)));

    STDERR("execute kernel 1...");
    state = state.apply(ExecuteOp(gkernel1));

    decisions = state.get_decisions(plat);
    for (const auto &dp : decisions) {
      std::cerr << dp->desc() << "\n";
    }

  }


}