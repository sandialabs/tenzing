/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "sched/randomness.hpp"

#include "sched/numeric.hpp"

namespace randomness {

bool runs_test(const std::vector<double> &v) {
    // https://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm
    // H0: random
    // Ha: not random

    const double median = med(v);
    std::vector<bool> deltas;
    size_t n1=0, n2=0;
    for (double t : v) {
        if (t >= median) {
            deltas.push_back(1);
            ++n1;
        } else {
            deltas.push_back(0);
            ++n2;
        }
    }

    if (n1 < 10 || n2 < 10) {
        return true; // does not apply
    }

    size_t nRuns = 1;
    for (size_t i = 0; i < deltas.size() - 1; ++i) {
        if (deltas[i] != deltas[i+1]) {
            ++nRuns;
        }
    }

    double R_bar = 2*n1*n2 / double(n1 + n2) + 1;
    double s = std::sqrt( 
        2*n1*n2*(2*n1*n2-n1-n2) 
        / 
        double((n1+n2)*(n1+n2)*(n1+n2-1)) 
    );

    double Z = std::abs((double(nRuns) - R_bar) / s);
    // STDERR("Z=" << Z);
    // standard normal table
    if (Z > 1.96) { // a=0.05, 5% chance of rejecting a true random
    // if (Z > 1.645) { // a=0.10
    // if (Z > 1.282) { // a=0.20
        return true;
    } else {
        return false;
    }
}

bool compound_test(const std::vector<double> &v) {
    if (runs_test(v)) return true;
    return false;
}

} // namespace randomness