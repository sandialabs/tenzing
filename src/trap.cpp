#include "sched/trap.hpp"

#include <iostream>

// what the SIGINT handler will call
std::function<void(int)> global_handler;
void int_caller(int signal) {
    std::cerr << "CAUGHT " << signal << "\n";
    unregister_handler();
    global_handler(signal);
    exit(1);
}

void abrt_caller(int signal) {
    std::cerr << "CAUGHT " << signal << "\n";
    unregister_handler();
    global_handler(signal);
    exit(1);
}

void register_handler(std::function<void(int)> func) {
    global_handler = func;
    signal(SIGINT, int_caller);
    signal(SIGABRT, abrt_caller);
}

void unregister_handler() {
    signal(SIGABRT, SIG_DFL);
    signal(SIGINT, SIG_DFL);
}
