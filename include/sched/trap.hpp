#include <signal.h>

#include <functional>

typedef void(Handler)(int);

void register_handler(std::function<void(int)> func);

void unregister_handler();

