# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
# terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
# software.


function(add_csv tgt)
    add_executable(${tgt} ${ARGN})
    target_link_libraries(${tgt} tenzing)
    set_standards(${tgt})
    set_options(${tgt})
    target_link_options(${tgt} PUBLIC -rdynamic)
endfunction()

function(add_spmv tgt)
    add_executable(${tgt} ${ARGN})
    target_link_libraries(${tgt} tenzing)
    set_standards(${tgt})
    set_options(${tgt})
    target_include_directories(${tgt} SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty)
    target_include_directories(${tgt} SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/cwpearson)
    target_link_options(${tgt} PUBLIC -rdynamic)
endfunction()

# add_csv(mcts-csv-avg-time mcts_csv_avg_time.cu)
# add_csv(mcts-csv-coverage mcts_csv_coverage.cu)
# add_csv(mcts-csv-min-time mcts_csv_min_time.cu)
# add_csv(mcts-csv-random mcts_csv_random.cu)

add_spmv(platform-mcts-random platform_mcts_random.cu)
add_spmv(platform-mcts-min-time platform_mcts_min_time.cu)
add_spmv(platform-mcts-coverage platform_mcts_coverage.cu)
add_spmv(spmv-brute spmv_brute.cu)
