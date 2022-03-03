/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include <cstdlib>
#include <iostream>

#include "tenzing/init.hpp"

#include "tenzing/reproduce.hpp"

namespace init {
constexpr const char *CONTACT_NOTICE = "For questions, contact Carl Pearson <cwpears@sandia.gov>.";
constexpr const char *RESEARCH_NOTICE =
    "This is research software. It may not work correctly, or at all.";
constexpr const char *COPYRIGHT_NOTICE =
    "Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the "
    "terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this "
    "software";

/* singleton for the global library configuration
 */
class Config {

public:
  static Config &get_instance() {
    static Config instance;
    return instance;
  }

  bool printNotice; // whether notices should be prinited on init()

private:
  Config() : printNotice(true) {
    char *acked = std::getenv("TENZING_ACK_NOTICE");
    if (acked) {
      printNotice = false;
    }
  }
};

void maybe_print_notices_and_exit() {
  if (Config::get_instance().printNotice) {
    std::cerr << tenzing::reproduce::version_string() << std::endl;
    std::cerr << CONTACT_NOTICE << std::endl;
    std::cerr << RESEARCH_NOTICE << std::endl;
    std::cerr << COPYRIGHT_NOTICE << std::endl;
    std::cerr << std::endl;
    std::cerr << " ---> Define TENZING_ACK_NOTICE in your environment to silence  <---\n";
    std::cerr << " ---> Define TENZING_ACK_NOTICE in your environment to silence  <---\n";
    std::cerr << " ---> Define TENZING_ACK_NOTICE in your environment to silence  <---\n";
    exit(1);
  }
}

} // namespace init

namespace tenzing {
void init(int argc, char **argv) {
  static bool inited = false;
  if (inited)
    return;
  if (!inited)
    inited = true;
  init::maybe_print_notices_and_exit();
}
} // namespace tenzing