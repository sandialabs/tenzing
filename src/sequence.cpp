/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "sched/sequence.hpp"

#include "sched/operation_serdes.hpp"
#include "sched/cuda/ops_cuda.hpp"

#include <sstream>



std::string Equivalence::str() const {
  std::stringstream ss;

  ss << "streams: {" << streams_.str() << "} ";
  ss << "events: {" << events_.str() << "}";
  return ss.str();
}

Equivalence get_equivalence(const Sequence<BoundOp> &a, const Sequence<BoundOp> &b) {
  if (a.size() != b.size()) {
    return Equivalence::falsy();
  }

  Equivalence eq;
  auto ai = a.begin();
  auto bi = b.begin();

  // just check first part of b
  for (; ai < a.end(); ++ai, ++bi) {
    if ((*ai)->name() == (*bi)->name()) {

      { // check stream bijection
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
              if (!eq.check_or_insert(ass[i], bss[i])) { // false if bijection is broken
                return Equivalence::falsy();
              }
            }
          }
        } else { // false if both operations don't have streams
          return Equivalence::falsy();
        }
      }

      { // event bijection
        auto ae = std::dynamic_pointer_cast<HasEvent>(*ai);
        auto be = std::dynamic_pointer_cast<HasEvent>(*bi);

        if (bool(ae) == bool(be)) {
          if (ae && be) {
            auto aee = ae->get_events();
            auto bee = be->get_events();
            if (aee.size() != bee.size()) { // false if different numbers of events
              return Equivalence::falsy();
            }

            for (size_t i = 0; i < aee.size(); ++i) {
              if (!eq.check_or_insert(aee[i], bee[i])) { // false if bijection is broken
                return Equivalence::falsy();
              }
            }
          }
        } else { // false if both operations don't have events
          return Equivalence::falsy();
        }
      }
    } else { // falsy if operation names are different
      return Equivalence::falsy();
    }
  }
  return eq;


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
    Sequence<BoundOp> seq;
    from_json(des, g, seq);
    return seq;
  } else {
    return order;
  }
}


std::string get_desc_delim(const Sequence<BoundOp> &seq, const std::string &delim) {
  std::string s;

  for (auto si = seq.begin(); si < seq.end(); ++si) {
    s += (*si)->desc();
    if (si + 1 < seq.end()) {
      s += delim;
    }
  } 

  return s;
}

template<>
Sequence<BoundOp>::const_iterator
Sequence<BoundOp>::find_unbound(const std::shared_ptr<OpBase> &e) const {

  // unbound version if bound
  std::shared_ptr<OpBase> ue;
  if (auto bgo = std::dynamic_pointer_cast<BoundGpuOp>(e)) {
    ue = bgo->unbound();
  } else {
    ue = e;
  }

  for (auto it = ops_.begin(); it < ops_.end(); ++it) {

    // get unbound version if bound
    std::shared_ptr<OpBase> uve;
    if (auto bgo = std::dynamic_pointer_cast<BoundGpuOp>(*it)) {
      uve = bgo->unbound();
    } else {
      uve = *it;
    }

    if (uve->eq(ue)) {
      return it;
    }
  }
  return ops_.end();
}