/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include <vector>

namespace randomness {

// true if rejected
bool runs_test(const std::vector<double> &v);

// runs many tests and rejects if any fail
bool compound_test(const std::vector<double> &v);

}