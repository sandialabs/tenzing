/*! contains the doctest main function if testing is enabled
*/

#include <doctest/doctest.hpp>

int main(int argc, char** argv) {
    doctest::Context context;
    context.applyCommandLine(argc, argv);

    int res = context.run(); // run

    if(context.shouldExit()) // important - query flags (and --exit) rely on the user doing this
        return res;          // propagate the result of the tests
    
 
    return res; // the result from doctest is propagated here as well
}
