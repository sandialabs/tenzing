/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include "macro_at.hpp"

#include <vector>
#include <cstdint>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#include <cmath>
#pragma GCC diagnostic pop
#include <algorithm>

template <typename T>
double avg(const std::vector<T> &v) {
    double acc = 0;
    for (const T &e : v) {
        acc += double(e);
    }
    return acc / v.size();
}

template <typename T>
double med(const std::vector<T> &v) {
    std::vector<T> vv(v);
    std::sort(vv.begin(), vv.end());
    if (vv.size() % 2) {
        return vv[vv.size() / 2];
    } else {
        return (vv[vv.size() / 2] + vv[vv.size() / 2 + 1]) / 2.0;
    }
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

template <typename T>
double corr(const std::vector<T> &a, const std::vector<T> &b) {


#if 0
    // convert to log(x+1)
    std::vector<double> a(_a.size());
    std::vector<double> b(_b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = std::log(_a[i]+1.0);
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = std::log(_b[i]+1.0);
    }
#endif

    if (a.size() != b.size()) {
        THROW_RUNTIME("vectors must be same size");
    }

    if (0 == a.size()) return 0;

    const double aBar = avg(a);
    const double bBar = avg(b);
    const double aS = stddev(a);
    const double bS = stddev(b);
    
    double acc = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      acc += (double(a[i]) - aBar) * (double(b[i]) - bBar);
    }
    double c = acc / (a.size() * aS * bS);
    if (c < -1.01) {
        THROW_RUNTIME(
            "corr=" << c << " < -1:" 
            << " aBar=" << aBar 
            << " bBar=" << bBar 
            << " stddev(a)=" << aS 
            << " stddev(b)=" << bS
        );
    }
    if (c > 1.01) {
        THROW_RUNTIME(
            "corr=" << c << " > 1:" 
            << " aBar=" << aBar 
            << " bBar=" << bBar 
            << " stddev(a)=" << aS
            << " stddev(b)-" << bS
        );
    }
    // fix any numerical weirdness
    if (c > 1) c = 1;
    if (c < -1) c = -1;
    return c;
}

template<typename T> std::vector<T> prime_factors(T n);
template<typename T> T round_up(T x, T step);