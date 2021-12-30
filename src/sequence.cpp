#include "sched/sequence.hpp"

#include "sched/operation_serdes.hpp"
#include "sched/ops_cuda.hpp"

bool Equivalence::check_or_insert(const Stream &a, const Stream &b) {
#warning should both always be inserted?
  auto ab = sAtoB.insert(std::make_pair(a, b));
  auto ba = sBtoA.insert(std::make_pair(b, a));
  if (ab.second && ba.second) { // both inserted
    return true;
  } else { // bijection must hold
    return ab.first->second == b && ba.first->second == a;
  }
}

Equivalence get_equivalence(const Sequence<BoundOp> &a, const Sequence<BoundOp> &b) {

  // no equivalence possible with size mismatch
  if (a.size() != b.size()) {
    return Equivalence::falsy();
  }

  Equivalence eq;
  auto ai = a.begin();
  auto bi = b.begin();

  for (; ai < a.end() && bi < b.end(); ++ai, ++bi) {
    if ((*ai)->name() == (*bi)->name()) {

      auto as = std::dynamic_pointer_cast<HasStream>(*ai);
      auto bs = std::dynamic_pointer_cast<HasStream>(*bi);

      if (bool(as) == bool(bs)) {
        if (as && bs) {
          auto ass = as->get_streams();
          auto bss = bs->get_streams();
          if (ass.size() != bss.size()) { // false if different numbers of streams
            return Equivalence::falsy();
          }

          for (size_t i = 0; i < ass.size(); ++i) {
            if (!eq.check_or_insert(ass[i], bss[i])) { // false if no equivalence
              return Equivalence::falsy();
            }
          }
        }
      } else { // false if both operations don't have streams
        return Equivalence::falsy();
      }

    } else { // falsy if operation names are different
      return Equivalence::falsy();
    }
  }

  #warning unfinished
}

Sequence<BoundOp> mpi_bcast(const Sequence<BoundOp> &order, const Graph<OpBase> &g, MPI_Comm comm) {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string jsonStr;

  // serialize the sequence to json
  if (0 == rank) {
    nlohmann::json json;
    to_json(json, order, g);
    jsonStr = json.dump();
    STDERR(jsonStr);
  }

  // broadcast the JSON length and resize the receiving string
  {
    size_t jsonStrSz = jsonStr.size();
    MPI_Bcast(&jsonStrSz, sizeof(jsonStrSz), MPI_BYTE, 0, comm);
    jsonStr.resize(jsonStrSz);
  }

  // broadcast the JSON
  MPI_Bcast(&jsonStr[0], jsonStr.size(), MPI_CHAR, 0, comm);

  if (0 != rank) {
    // turn json string into json
    nlohmann::json des = nlohmann::json::parse(jsonStr);

    // deserialize the string into a sequence
    std::vector<std::shared_ptr<BoundOp>> seq;
    from_json(des, g, seq);
    return seq;
  } else {
    return order;
  }
}