/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include <signal.h>

#include <functional>

typedef void(Handler)(int);

void register_handler(std::function<void(int)> func);

void unregister_handler();

