# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
# terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
# software.

add_executable(halo-mcts-min-time halo_min_time.cu)
target_link_libraries(halo-mcts-min-time halo-exchange)
set_standards(halo-mcts-min-time)
set_options(halo-mcts-min-time)
target_link_options(halo-mcts-min-time PUBLIC -rdynamic)

add_executable(halo-mcts-coverage halo_min_time.cu)
target_link_libraries(halo-mcts-coverage halo-exchange)
set_standards(halo-mcts-coverage)
set_options(halo-mcts-coverage)
target_link_options(halo-mcts-coverage PUBLIC -rdynamic)

# add_executable(mcts-halo-anticorr halo_anticorr.cu )
# target_link_libraries(mcts-halo-anticorr sched)
# target_compile_options(mcts-halo-anticorr PRIVATE -Xcompiler=-fno-pie)
# target_compile_options(mcts-halo-anticorr PRIVATE -Xcompiler=-no-pie)
# set_standards(mcts-halo-anticorr)
# set_options(mcts-halo-anticorr)
# target_link_options(mcts-halo-anticorr PUBLIC -rdynamic)

# add_executable(mcts-halo-balance halo_balance.cu )
# target_link_libraries(mcts-halo-balance sched)
# target_compile_options(mcts-halo-balance PRIVATE -Xcompiler=-fno-pie)
# target_compile_options(mcts-halo-balance PRIVATE -Xcompiler=-no-pie)
# set_standards(mcts-halo-balance)
# set_options(mcts-halo-balance)
# target_link_options(mcts-halo-balance PUBLIC -rdynamic)