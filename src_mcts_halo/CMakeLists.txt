add_executable(mcts-halo halo.cu ops_halo_exchange.cu)
target_link_libraries(mcts-halo sched)
target_compile_options(mcts-halo PRIVATE -Xcompiler=-fno-pie)
target_compile_options(mcts-halo PRIVATE -Xcompiler=-no-pie)
set_standards(mcts-halo)
set_options(mcts-halo)

# function names in back trace
target_link_options(mcts-halo PUBLIC -rdynamic)

