add_library(sched
schedule.cpp
graph.cpp
operation.cpp
ops_mpi.cpp
)

set_standards(sched)
set_options(sched)
target_compile_definitions(sched PRIVATE VIEW_CHECK_BOUNDS)
target_compile_definitions(sched PRIVATE SANITY_CHECKS)
target_include_directories(sched PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sched MPI::MPI_CXX)
target_link_libraries(sched CUDA::cudart)
target_link_libraries(sched CUDA::cusparse)