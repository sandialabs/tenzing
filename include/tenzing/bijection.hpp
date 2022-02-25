#pragma once

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
