/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

/*! \file Useful reproducibility utilities
 */

#include <iostream>
#include <vector>
#include <sstream>

#include <nlohmann/json.hpp>

#include "tenzing/reproduce.hpp"

#include "tenzing/version.hpp"

namespace tenzing {
namespace reproduce {

void dump_with_cli(int argc, char **argv) {
  nlohmann::json j;
  j["major"] = TENZING_VERSION_MAJOR;
  j["minor"] = TENZING_VERSION_MINOR;
  j["patch"] = TENZING_VERSION_PATCH;
  j["hash"] = TENZING_VERSION_HASH;

  std::vector<const char *> args;
  if (nullptr != argv) {
    for (int i = 0; i < argc; ++i) {
      args.push_back(argv[i]);
    }
  }
  j["args"] = args;
  std::cout << j.dump() << std::endl;
}

std::string version_string() {

  std::stringstream ss;

  ss << "tenzing " << TENZING_VERSION_MAJOR << "." << TENZING_VERSION_MINOR << "." << TENZING_VERSION_PATCH << "-"
     << TENZING_VERSION_HASH << std::endl;
  return ss.str();
}

} // namespace reproduce
} // namespace tenzing