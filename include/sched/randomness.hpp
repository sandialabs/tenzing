#pragma once

#include <vector>

namespace randomness {

// true if rejected
bool runs_test(const std::vector<double> &v);

// runs many tests and rejects if any fail
bool compound_test(const std::vector<double> &v);

}