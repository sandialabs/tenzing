# 3.17+ for CMAKE_CUDA_KNOWN_FEATURES/cuda_std_11
# 3.18+ for CUDA_ARCHITECTURES
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(sched LANGUAGES CXX CUDA VERSION 0.1.0.0)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting to OFF")
    set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "")
endif()


find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

message(STATUS "MPI_VERSION:                 ${MPI_VERSION}")
message(STATUS "MPI_CXX_COMPILER:            ${MPI_CXX_COMPILER}")
message(STATUS "MPI_CXX_COMPILE_OPTIONS:     ${MPI_CXX_COMPILE_OPTIONS}")
message(STATUS "MPI_CXX_COMPILE_DEFINITIONS: ${MPI_CXX_COMPILE_DEFINITIONS}")
message(STATUS "MPI_CXX_INCLUDE_DIRS:        ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "MPI_CXX_LIBRARIES:           ${MPI_CXX_LIBRARIES}")
message(STATUS "MPI_CXX_LINK_FLAGS:          ${MPI_CXX_LINK_FLAGS}")
message(STATUS "MPIEXEC_EXECUTABLE:          ${MPIEXEC_EXECUTABLE}")
message(STATUS "MPIEXEC_NUMPROC_FLAG:        ${MPIEXEC_NUMPROC_FLAG}")
message(STATUS "MPIEXEC_MAX_NUMPROCS:        ${MPIEXEC_MAX_NUMPROCS}")
message(STATUS "MPIEXEC_PREFLAGS:            ${MPIEXEC_PREFLAGS}")
message(STATUS "MPIEXEC_POSTFLAGS:           ${MPIEXEC_POSTFLAGS}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES:    ${CMAKE_CUDA_ARCHITECTURES}")


function(set_options target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
    -Wall
    -Wextra
    -Wcast-align;
    -Wdisabled-optimization;
    -Wformat=2;
    -Winit-self;
    -Wlogical-op;
    -Wmissing-include-dirs;
    -Woverloaded-virtual;
    -Wpointer-arith;
    -Wshadow;
    -Wstrict-aliasing;
    -Wswitch-enum;
    -Wvla;
    >
  )
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    --Wno-deprecated-gpu-targets;
    --expt-extended-lambda;
    -Xcompiler=-Wall;
    -Xcompiler=-Wextra;
    -Xcompiler=-Wcast-align;
    -Xcompiler=-Wdisabled-optimization;
    -Xcompiler=-Wformat=2;
    -Xcompiler=-Winit-self;
    -Xcompiler=-Wlogical-op;
    -Xcompiler=-Wmissing-include-dirs;
    -Xcompiler=-Woverloaded-virtual;
    -Xcompiler=-Wpointer-arith;
    -Xcompiler=-Wshadow;
    -Xcompiler=-Wstrict-aliasing;
    -Xcompiler=-Wswitch-enum;
    -Xcompiler=-Wvla;
    -Xptxas=-v;
    -lineinfo;
    >
  )
endfunction()

function(set_standards target)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD 11)
    set_property(TARGET ${target} PROPERTY CXX_EXTENSIONS OFF)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET ${target} PROPERTY CUDA_STANDARD 11)
    set_property(TARGET ${target} PROPERTY CUDA_STANDARD_REQUIRED ON)
endfunction()


add_executable(main main.cu schedule.cpp graph.cpp operation.cpp)
set_standards(main)
set_options(main)

# link MPI
target_link_libraries(main MPI::MPI_CXX)

# link CUDA
target_link_libraries(main CUDA::cudart)
target_link_libraries(main CUDA::cusparse)


target_compile_definitions(main PRIVATE VIEW_CHECK_BOUNDS)
target_compile_definitions(main PRIVATE SANITY_CHECKS)
target_include_directories(main SYSTEM PRIVATE thirdparty)