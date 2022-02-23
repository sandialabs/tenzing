# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
# terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
# software.


# 3.17+ for CMAKE_CUDA_KNOWN_FEATURES/cuda_std_11
# 3.18+ for CUDA_ARCHITECTURES
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules/")

project(sched LANGUAGES CXX CUDA VERSION 0.1.0.0)

include(GetGitRevisionDescription)
git_local_changes(SCHED_LOCAL_CHANGES)
get_git_head_revision(SCHED_REFSPEC SCHED_GIT_HASH)
git_describe(SCHED_GIT_VERSION --tags)

message(STATUS "SCHED_LOCAL_CHANGES: ${SCHED_LOCAL_CHANGES}")
message(STATUS "SCHED_REFSPEC:       ${SCHED_REFSPEC}")
message(STATUS "SCHED_GIT_HASH:      ${SCHED_GIT_HASH}")
message(STATUS "SCHED_GIT_VERSION:   ${SCHED_GIT_VERSION}")

string(REGEX REPLACE "^v([0-9]+)\\..*" "\\1" SCHED_GIT_MAJOR "${SCHED_GIT_VERSION}")
string(REGEX REPLACE "^v[0-9]+\\.([0-9]+).*" "\\1" SCHED_GIT_MINOR "${SCHED_GIT_VERSION}")
string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" SCHED_GIT_PATCH "${SCHED_GIT_VERSION}")
# string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.[0-9]+(.*)" "\\1" SCHED_GIT_HASH "${SCHED_GIT_VERSION}")

message(STATUS "SCHED_GIT_MAJOR:     ${SCHED_GIT_MAJOR}")
message(STATUS "SCHED_GIT_MINOR:     ${SCHED_GIT_MINOR}")
message(STATUS "SCHED_GIT_PATCH:     ${SCHED_GIT_PATCH}")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting to OFF")
    set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "")
endif()

if(NOT DEFINED CMAKE_BUILD_TYPE OR NOT CMAKE_BUILD_TYPE)
    message(STATUS "CMAKE_BUILD_TYPE not defined, setting to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build" FORCE)
endif()

message(STATUS "CMAKE_BUILD_TYPE:            ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION:  ${CMAKE_CXX_COMPILER_VERSION}")
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(NUMA)
message(STATUS "NUMA_LIBRARIES:              ${NUMA_LIBRARIES}")
message(STATUS "NUMA_INCLUDE_DIRS:           ${NUMA_INCLUDE_DIRS}")
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

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mcpu=native" CXX_HAS_MCPU)
check_cxx_compiler_flag("-march=native" CXX_HAS_MARCH)


function(set_options target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
    -Wall;
    -Wextra;
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
  if (${CXX_HAS_MARCH})
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
  elseif (${CXX_HAS_MCPU})
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mcpu=native;-mtune=native>)
  endif()

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
  if (${CXX_HAS_MARCH})
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>)
  elseif (${CXX_HAS_MCPU})
    target_compile_options(${target} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-mcpu=native;
    -Xcompiler=-mtune=native
    >)
  endif()


  ## GCC 7.5.0 can't use pragma GCC diagnostic ignore "-Wunknown-pragmas" to ignore unknown pragmas
  if (CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 7.5.0 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:
      -Wno-unknown-pragmas;
      >
    )
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
      -Xcompiler=-Wno-unknown-pragmas;
      >
    )
  endif()

endfunction()

function(set_standards target)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD 11)
    set_property(TARGET ${target} PROPERTY CXX_EXTENSIONS OFF)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET ${target} PROPERTY CUDA_STANDARD 11)
    set_property(TARGET ${target} PROPERTY CUDA_STANDARD_REQUIRED ON)
endfunction()

add_subdirectory(src)
add_subdirectory(src_mcts_halo)
# add_subdirectory(src_mcts_test)
add_subdirectory(src_spmv)

