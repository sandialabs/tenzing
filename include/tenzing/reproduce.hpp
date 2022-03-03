/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include <string>
#include <vector>


namespace tenzing {
namespace reproduce {

void dump_with_cli(int argc, char **argv);
std::string version_string();

} // namespace reproduce
} // namespace tenzing