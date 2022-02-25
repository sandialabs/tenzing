/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include <memory>
#include <sstream>

namespace tenzing {

// dynamic_pointer_cast U to T, or throw
template <typename T, typename U>
std::shared_ptr<T> cast_or_throw(const std::shared_ptr<U> &u, const char *file, int line) {
  auto t = std::dynamic_pointer_cast<T>(u);
  if (!t) {
    std::stringstream ss;
    ss << file << ":" << line << ": "
       << "failed pointer cast"
       << "\n";
    throw std::runtime_error(ss.str());
  }
  return t;
}
} // namespace tenzing

#define TENZING_MUST_CAST(T, x) tenzing::cast_or_throw<T>(x, __FILE__, __LINE__)

template <typename T, typename U> bool isa(const std::shared_ptr<U> &ptr) {
  return bool(std::dynamic_pointer_cast<T>(ptr));
}