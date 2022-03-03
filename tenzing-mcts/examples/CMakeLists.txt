# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
# terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
# software.

function(add_csv tgt)
    add_executable(${tgt} ${ARGN})
    target_link_libraries(${tgt} tenzing-mcts)
    tenzing_set_standards(${tgt})
    tenzing_set_options(${tgt})
    target_link_options(${tgt} PUBLIC -rdynamic)
endfunction()

function(add_spmv tgt)
    add_executable(${tgt} ${ARGN})
    target_link_libraries(${tgt} tenzing-mcts)
    tenzing_set_standards(${tgt})
    tenzing_set_options(${tgt})
    target_link_options(${tgt} PUBLIC -rdynamic)
endfunction()

add_executable(halo-mcts-min-time halo_min_time.cu)
target_link_libraries(halo-mcts-min-time tenzing-mcts)
tenzing_set_standards(halo-mcts-min-time)
tenzing_set_options(halo-mcts-min-time)
target_link_options(halo-mcts-min-time PUBLIC -rdynamic)

add_executable(halo-mcts-coverage halo_coverage.cu)
target_link_libraries(halo-mcts-coverage tenzing-mcts)
tenzing_set_standards(halo-mcts-coverage)
tenzing_set_options(halo-mcts-coverage)
target_link_options(halo-mcts-coverage PUBLIC -rdynamic)

# add_csv(mcts-csv-avg-time mcts_csv_avg_time.cu)
# add_csv(mcts-csv-coverage mcts_csv_coverage.cu)
# add_csv(mcts-csv-min-time mcts_csv_min_time.cu)
# add_csv(mcts-csv-random mcts_csv_random.cu)

add_spmv(spmv-random   spmv_random.cu)
add_spmv(spmv-min-time spmv_min_time.cu)
add_spmv(spmv-coverage spmv_coverage.cu)
