/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "tenzing/numeric.hpp"

#include <algorithm>


template<typename T>
std::vector<T> prime_factors(T n) {
  std::vector<T> result;
  if (0 == n) {
    return result;
  }
  while (n % 2 == 0) {
    result.push_back(2);
    n = n / 2;
  }
  for (int i = 3; i <= sqrt(n); i = i + 2) {
    while (n % i == 0) {
      result.push_back(i);
      n = n / i;
    }
  }
  if (n > 2)
    result.push_back(n);
  std::sort(result.begin(), result.end(), [](T a, T b) { return b < a; });
  return result;
}

template std::vector<int> prime_factors(int n);

template<typename T>
T round_up(T n, T step) {
  return (n + step - 1) / step * step;
}

#define INST_ROUND_UP(T) template T round_up(T n, T step)
INST_ROUND_UP(size_t);
INST_ROUND_UP(int);
