add_executable(halo-mcts-min-time halo_min_time.cu pattern_halo_exchange.cu)
target_link_libraries(halo-mcts-min-time sched)
set_standards(halo-mcts-min-time)
set_options(halo-mcts-min-time)

# function names in back trace
target_link_options(halo-mcts-min-time PUBLIC -rdynamic)


# add_executable(mcts-halo-coverage halo_coverage.cu )
# target_link_libraries(mcts-halo-coverage sched)
# target_compile_options(mcts-halo-coverage PRIVATE -Xcompiler=-fno-pie)
# target_compile_options(mcts-halo-coverage PRIVATE -Xcompiler=-no-pie)
# set_standards(mcts-halo-coverage)
# set_options(mcts-halo-coverage)
# target_link_options(mcts-halo-coverage PUBLIC -rdynamic)

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