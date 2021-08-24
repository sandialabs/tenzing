#pragma once

#include <sstream>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__) 

#define THROW_RUNTIME(msg) \
{\
    std::stringstream ss;\
    ss << __FILE__ << ":" << __LINE__ << ": \"" << msg << "\"\n";\
}
