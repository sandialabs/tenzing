/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include <map>
#include <memory>
#include <vector>

#include "sched/graph.hpp"
#include "sched/operation.hpp"

template <typename OpType> using Sequence = std::vector<std::shared_ptr<OpType>>;

template <typename OP> Event new_unique_event(const Sequence<OP> &seq) {
  std::set<Event> taken;

  for (const auto &op : seq) {
    if (auto he = std::dynamic_pointer_cast<HasEvent>(op)) {
      for (const Event &event : he->get_events()) {
        taken.insert(event);
      }
    }
  }

  for (Event::id_t id = 0; true; ++id) {
    if (0 == taken.count(Event(id))) {
      return Event(id);
    }
  }
}

template <typename T> class Bijection {
  std::map<T, T> map_;

public:
  bool check_or_insert(const T &a, const T &b) {

    // STDERR("look up " << a << " -> " << b);
    const size_t ca = map_.count(a);
    const size_t cb = map_.count(b);

    // does not contain
    if (0 == ca && 0 == cb) {
      //   STDERR("insert " << a << " <-> " << b);
      map_.insert(std::make_pair(a, b));
      map_.insert(std::make_pair(b, a));
      return true;
    } else if (0 != ca && 0 != cb) {
      //   STDERR("check " << b << " <-> " << a);
      return map_.at(b) == a && map_.at(a) == b;
    } else {
      return false;
    }
  }

  bool empty() const { return map_.empty(); }

  std::string str() const {
    std::stringstream ss;

    for (auto kvi = map_.begin(); kvi != map_.end(); ++kvi) {
      ss << kvi->first << "->" << kvi->second;
      {
        auto next = kvi;
        next++;
        if (next != map_.end()) {
          ss << ",";
        }
      }
    }

    return ss.str();
  }
};

class Equivalence {
  Bijection<Stream> streams_;
  Bijection<Event> events_;

public:
  // true if some bijection of streams and events renders two sequences equal
  operator bool() const { return !streams_.empty() || !events_.empty(); }

  // if map[a] does not exist, insert map[a] = b and return true
  // else, return map[a] == b
  bool check_or_insert(const Stream &a, const Stream &b) { return streams_.check_or_insert(a, b); }
  bool check_or_insert(const Event &a, const Event &b) { return events_.check_or_insert(a, b); }

  std::string str() const;

  static Equivalence falsy() { return Equivalence(); }
};

// try to discover an equivalence between two sequences.
// if not, return falsy
Equivalence get_equivalence(const Sequence<BoundOp> &a, const Sequence<BoundOp> &b);

/* broadcast `order` from rank 0 to the other ranks
 */
Sequence<BoundOp> mpi_bcast(const Sequence<BoundOp> &order, const Graph<OpBase> &g, MPI_Comm comm);

// string of BoundOp->desc() separated by delim
std::string get_desc_delim(const Sequence<BoundOp> &seq, const std::string &delim);
