#include <cstdlib>
#include <iostream>

#include "sched/init.hpp"
#include "sched/version.hpp"

namespace init {
constexpr const char *VERSION_NOTICE = "sched " SCHED_XSTR(SCHED_VERSION_MAJOR) "." SCHED_XSTR(
    SCHED_VERSION_MINOR) "." SCHED_XSTR(SCHED_VERSION_PATCH);
constexpr const char *CONTACT_NOTICE = "For questions, contact Carl Pearson <cwpears@sandia.gov>.";
constexpr const char *RESEARCH_NOTICE =
    "This is research software. It may not work correctly, or at all.";
constexpr const char *COPYRIGHT_NOTICE =
    "Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC "
    "(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, there is a "
    "non-exclusive license for use of this work by or on behalf of the U.S. "
    "Government. Export of this data may require a license from the United "
    "States Government.";

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
    char *acked = std::getenv("SCHED_ACK_NOTICE");
    if (acked) {
      printNotice = false;
    }
  }
};

void maybe_print_notices_and_exit() {
  if (Config::get_instance().printNotice) {
    std::cerr << VERSION_NOTICE << std::endl;
    std::cerr << CONTACT_NOTICE << std::endl;
    std::cerr << RESEARCH_NOTICE << std::endl;
    std::cerr << COPYRIGHT_NOTICE << std::endl;
    std::cerr << std::endl;
    std::cerr << "Set SCHED_ACK_NOTICE to silence\n";
    exit(1);
  }
}

} // namespace init

namespace sched {
void init() {
  static bool inited = false;
  if (inited)
    return;
  if (!inited)
    inited = true;
  init::maybe_print_notices_and_exit();
}
} // namespace sched