# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
# terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
# software.

# add an object library for integration with doctest, static library may remove test registration code
add_library(tenzing-object OBJECT
benchmarker.cpp
brute.cpp
counters.cpp
event_synchronizer.cpp
graph.cpp
init.cpp
mcts_node.cpp
mcts.cpp
numa.cpp
numeric.cpp
operation_serdes.cpp
operation.cpp
platform.cpp
randomness.cpp
schedule.cpp
sequence.cpp
state.cpp
test_impl.cpp
trap.cpp
cuda/ops_cuda.cpp
mpi/ops_mpi.cpp
)

set_standards(tenzing-object)
set_options(tenzing-object)
target_compile_definitions(tenzing-object PRIVATE VIEW_CHECK_BOUNDS)
target_compile_definitions(tenzing-object PRIVATE SANITY_CHECKS)
target_include_directories(tenzing-object PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(tenzing-object SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty)
target_include_directories(tenzing-object SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/cwpearson)
target_link_libraries(tenzing-object MPI::MPI_CXX)
target_link_libraries(tenzing-object CUDA::cudart)
target_link_libraries(tenzing-object CUDA::cusparse)
if (TENZING_ENABLE_TESTS)
    target_compile_definitions(tenzing-object PRIVATE TENZING_ENABLE_TESTS=1)
endif()

# create the version.hpp file
message(STATUS ${CMAKE_SOURCE_DIR}/include/sched/version.hpp.in -> ${CMAKE_BINARY_DIR}/include/sched/version.hpp)
configure_file(${CMAKE_SOURCE_DIR}/include/sched/version.hpp.in
                ${CMAKE_BINARY_DIR}/include/sched/version.hpp)
target_include_directories(tenzing-object PUBLIC ${CMAKE_BINARY_DIR}/include)


if (${NUMA_FOUND})
    target_compile_definitions(tenzing-object PRIVATE SCHED_USE_NUMA)
    target_include_directories(tenzing-object PUBLIC ${NUMA_INCLUDE_DIRS})
    target_link_libraries(tenzing-object ${NUMA_LIBRARIES})
endif() # NUMA_FOUND

# actual library that non-test binaries should link against (use object library as the input)
add_library(sched $<TARGET_OBJECTS:tenzing-object>)

add_subdirectory(spmv)
add_subdirectory(halo_exchange)

