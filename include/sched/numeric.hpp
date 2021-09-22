#pragma once

#include <vector>
#include <cstdint>
#include <cmath>

template <typename T>
double avg(const std::vector<T> &v) {
    double acc = 0;
    for (const T &e : v) {
        acc += double(e);
    }
    return acc / v.size();
}

template <typename T>
double var(const std::vector<T> &v) {
    const double vbar = avg(v);
    double acc = 0;
    for (const T &e : v) {
        acc += (double(e) - vbar) * (double(e) - vbar);
    }
    return acc / v.size();
}

template <typename T>
double stddev(const std::vector<T> &v) {
    return std::sqrt(var(v));
}

template<typename T> std::vector<T> prime_factors(T n);
template<typename T> T round_up(T x, T step);