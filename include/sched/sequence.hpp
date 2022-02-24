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

// template <typename OpType> using Sequence = std::vector<std::shared_ptr<OpType>>;

/*! \brief Represents a sequence of operations in a program

     basically a std::vector<std::shared_ptr<OpType> with some extra stuff
*/
template <typename OpType> class Sequence {
public:
  typedef std::shared_ptr<OpType> value_type;
  typedef std::vector<value_type> vector_type;
  typedef typename vector_type::iterator iterator;
  typedef typename vector_type::const_iterator const_iterator;
  typedef typename vector_type::size_type size_type;
  typedef typename vector_type::reference reference;
  typedef typename vector_type::const_reference const_reference;

private:
  vector_type ops_;

public:
  Sequence() = default;
  Sequence(const Sequence &other) = default;
  Sequence(Sequence &&other) = default;
  Sequence(std::initializer_list<value_type> il) : ops_(il) {}

  Sequence &operator=(std::initializer_list<value_type> il) {
    ops_ = il;
    return *this;
  }
  Sequence &operator=(const Sequence &rhs) = default;

  /*! \brief true if Sequence contains e or an unbound version of e
  */
  bool contains_unbound(const std::shared_ptr<OpBase> &e) const {
    // unbound version if bound
    std::shared_ptr<OpBase> ue;
    if (auto bgo = std::dynamic_pointer_cast<BoundGpuOp>(e)) {
      ue = bgo->unbound();
    } else {
      ue = e;
    }

    for (const auto &ve : ops_) {

      // get unbound version if bound
      std::shared_ptr<OpBase> uve;
      if (auto bgo = std::dynamic_pointer_cast<BoundGpuOp>(ve)) {
        uve = bgo->unbound();
      } else {
        uve = ve;
      }

      if (uve->eq(ue)) {
        return true;
      }
    }
    return false;
  }

  /// \brief true iff e is in unbound ops_
  const_iterator find_unbound(const std::shared_ptr<OpBase> &e) const;

  Event new_unique_event() const {
    std::set<Event> taken;

    for (const auto &op : ops_) {
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

  const vector_type &vector() const {return ops_;}

  void clear() {ops_.clear();}
  iterator erase(const_iterator position) {return ops_.erase(position); }

  void push_back(const value_type &val) { ops_.push_back(val); }
  void push_back(value_type &&val) { ops_.push_back(val); }

  iterator begin() noexcept { return ops_.begin(); }
  const_iterator begin() const noexcept { return ops_.begin(); }
  iterator end() noexcept { return ops_.end(); }
  const_iterator end() const noexcept { return ops_.end(); }

  size_type size() const noexcept { return ops_.size(); }

  reference operator[](size_type n) {return ops_[n];}
  const_reference operator[](size_type n) const {return ops_[n];}
};

#if 0
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
#endif

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
