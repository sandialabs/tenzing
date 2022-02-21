# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
# terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
# software.

add_library(sched
benchmarker.cpp
brute.cpp
counters.cpp
graph.cpp
mcts_node.cpp
mcts.cpp
numa.cpp
numeric.cpp
operation_serdes.cpp
operation.cpp
ops_cuda.cpp
ops_mpi.cpp
platform.cpp
randomness.cpp
schedule.cpp
sequence.cpp
trap.cpp
)

set_standards(sched)
set_options(sched)
target_compile_definitions(sched PRIVATE VIEW_CHECK_BOUNDS)
target_compile_definitions(sched PRIVATE SANITY_CHECKS)
target_include_directories(sched PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(sched SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty)
target_include_directories(sched SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/cwpearson)
target_link_libraries(sched MPI::MPI_CXX)
target_link_libraries(sched CUDA::cudart)
target_link_libraries(sched CUDA::cusparse)

if (${NUMA_FOUND})
    target_compile_definitions(sched PRIVATE SCHED_USE_NUMA)
    target_include_directories(sched PUBLIC ${NUMA_INCLUDE_DIRS})
    target_link_libraries(sched ${NUMA_LIBRARIES})
endif() # NUMA_FOUND
