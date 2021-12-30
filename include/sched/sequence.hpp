#pragma once

#include <map>
#include <memory>
#include <vector>

#include "sched/graph.hpp"
#include "sched/operation.hpp"
#include "sched/platform.hpp"

template <typename OpType> using Sequence = std::vector<std::shared_ptr<OpType>>;

struct Equivalence {
  std::map<Stream, Stream> sAtoB;
  std::map<Stream, Stream> sBtoA;
  std::map<Event, Event> eMap;

  // true if some bijection of streams and events renders two sequences equal
  operator bool() const { return !sAtoB.empty() && !sBtoA.empty() && !eMap.empty(); }

  // if map[a] does not exist, insert map[a] = b and return true
  // else, return map[a] == b
  bool check_or_insert(const Stream &a, const Stream &b);

  static Equivalence falsy() { return Equivalence(); }
};

// try to discover an equivalence between two sequences.
// if not, return falsy
Equivalence get_equivalence(const Sequence<BoundOp> &a, const Sequence<BoundOp> &b);

/* broadcast `order` from rank 0 to the other ranks
 */
Sequence<BoundOp> mpi_bcast(const Sequence<BoundOp> &order, const Graph<OpBase> &g, MPI_Comm comm);
