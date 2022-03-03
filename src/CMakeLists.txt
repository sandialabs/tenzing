# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
# terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
# software.

function(tenzing_set_includes target)
    target_include_directories(${target} PUBLIC ${tenzing_SOURCE_DIR}/include)
    target_include_directories(${target} SYSTEM PUBLIC ${tenzing_SOURCE_DIR}/thirdparty)
    target_include_directories(${target} SYSTEM PUBLIC ${tenzing_SOURCE_DIR}/thirdparty/cwpearson)
    #because of configuring the version file
    target_include_directories(${target} PUBLIC ${CMAKE_BINARY_DIR}/include)
    if (${NUMA_FOUND})
        target_include_directories(${target} PUBLIC ${NUMA_INCLUDE_DIRS})
    endif() # NUMA_FOUND
endfunction()

function(tenzing_set_links target)
    target_link_libraries(${target} MPI::MPI_CXX)
    target_link_libraries(${target} CUDA::cudart)
    target_link_libraries(${target} CUDA::cusparse)
    if (${NUMA_FOUND})
        target_link_libraries(${target} ${NUMA_LIBRARIES})
    endif() # NUMA_FOUND
endfunction()

function(tenzing_set_definitions target)
    target_compile_definitions(${target} PUBLIC TENZING_RUNTIME_CHECK_BOUNDS)
    target_compile_definitions(${target} PUBLIC TENZING_RUNTIME_SANITY_CHECKS)
    if (${TENZING_ENABLE_COUNTERS})
        target_compile_definitions(${target} PUBLIC TENZING_ENABLE_COUNTERS)
    endif()
    if (${NUMA_FOUND})
        target_compile_definitions(${target} PUBLIC TENZING_USE_NUMA)
    endif() # NUMA_FOUND
endfunction()

function(propagate_property dst src prop)
    get_target_property(val ${src} ${prop})
    message(STATUS ${val})
    set_target_properties(${dst} PROPERTIES ${prop} ${val})
endfunction()

if(NOT CMAKE_PROPERTY_LIST)
    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)
    
    # Convert command output into a CMake list
    string(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    string(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
endif()
    
function(print_properties)
    message("CMAKE_PROPERTY_LIST = ${CMAKE_PROPERTY_LIST}")
endfunction()
    
function(print_target_properties target)
    if(NOT TARGET ${target})
      message(STATUS "There is no target named '${target}'")
      return()
    endif()

    foreach(property ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" property ${property})

        # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
        if(property STREQUAL "LOCATION" OR property MATCHES "^LOCATION_" OR property MATCHES "_LOCATION$")
            continue()
        endif()

        get_property(was_set TARGET ${target} PROPERTY ${property} SET)
        if(was_set)
            get_target_property(value ${target} ${property})
            message("${target} ${property} = ${value}")
        endif()
    endforeach()
endfunction()

# add an object library for integration with doctest, static library may remove test registration code
add_library(tenzing-object OBJECT
benchmarker.cpp
counters.cpp
event_synchronizer.cpp
graph.cpp
init.cpp
numa.cpp
numeric.cpp
operation_serdes.cpp
operation.cpp
platform.cpp
randomness.cpp
reproduce.cpp
schedule.cpp
sequence.cpp
state.cpp
test_impl.cpp
trap.cpp
cuda/ops_cuda.cpp
mpi/ops_mpi.cpp
halo_exchange/ops_halo_exchange.cu
spmv/ops_spmv.cu
)

tenzing_set_standards(tenzing-object)
tenzing_set_options(tenzing-object)
tenzing_set_definitions(tenzing-object)
tenzing_set_includes(tenzing-object)
tenzing_set_links(tenzing-object)
if (TENZING_ENABLE_TESTS)
    target_compile_definitions(tenzing-object PRIVATE TENZING_ENABLE_TESTS=1)
endif()



# actual library that non-test binaries should link against (use object library as the input)
# https://gitlab.kitware.com/cmake/cmake/-/issues/18090
# https://stackoverflow.com/questions/38832528/transitive-target-include-directories-on-object-libraries
# it seems that object libraries might only carry things transitively through target_link_library on object librarys
# so, propagate some properties to the static library as well
#  It would be cool to do something like this: target_include_directories(tenzing SYSTEM PUBLIC $<TARGET_PROPERTY:tenzing-object,PUBLIC_SYSTEM_INCLUDE_DIRECTORIES>)
# but cmake does not offer a PUBLIC_SYSTEM_INCLUDE_DIRECTORIES target, so just duplicate it all
add_library(tenzing $<TARGET_OBJECTS:tenzing-object>)
tenzing_set_standards(tenzing)
tenzing_set_options(tenzing)
tenzing_set_definitions(tenzing)
tenzing_set_includes(tenzing)
tenzing_set_links(tenzing)
