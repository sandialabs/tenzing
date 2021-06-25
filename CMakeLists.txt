# 3.17+ for CMAKE_CUDA_KNOWN_FEATURES/cuda_std_11
# 3.18+ for CUDA_ARCHITECTURES
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(sched LANGUAGES CXX VERSION 0.1.0.0)


include(CheckLanguage)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting to OFF")
    set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "")
endif()

check_language(CUDA)
find_package(MPI)
find_package(CUDAToolkit)

if (MPI_FOUND)
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
endif()


function(set_cxx_options target)
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
endfunction()

function(set_cuda_options target)
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
>
)
endfunction()

function(set_cxx_standard target)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD 11)
    set_property(TARGET ${target} PROPERTY CXX_EXTENSIONS OFF)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)
    if (CMAKE_CUDA_COMPILER)
        set_property(TARGET ${target} PROPERTY CUDA_STANDARD 11)
        set_property(TARGET ${target} PROPERTY CUDA_STANDARD_REQUIRED ON)
    endif()
endfunction()


add_executable(main main.cpp)
set_cxx_standard(main)
set_cxx_options(main)
if (MPI_FOUND)
  target_link_libraries(main MPI::MPI_CXX)
  target_compile_definitions(main PRIVATE REAL_MPI)
else()
endif()

if (CMAKE_CUDA_COMPILER)
  target_link_libraries(main CUDA::cudart)
  target_compile_definitions(main PRIVATE REAL_CUDA)
else()
endif()