/*! contains the implementation of the doctest library when tests are enabled, otherwise empty
*/

#if TENZING_ENABLE_TESTS == 1
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.hpp>
#endif