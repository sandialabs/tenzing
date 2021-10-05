add_executable(mcts-halo-fastest halo_fastest.cu ops_halo_exchange.cu)
target_link_libraries(mcts-halo-fastest sched)
target_compile_options(mcts-halo-fastest PRIVATE -Xcompiler=-fno-pie)
target_compile_options(mcts-halo-fastest PRIVATE -Xcompiler=-no-pie)
set_standards(mcts-halo-fastest)
set_options(mcts-halo-fastest)

# function names in back trace
target_link_options(mcts-halo-fastest PUBLIC -rdynamic)


add_executable(mcts-halo-coverage halo_coverage.cu ops_halo_exchange.cu)
target_link_libraries(mcts-halo-coverage sched)
target_compile_options(mcts-halo-coverage PRIVATE -Xcompiler=-fno-pie)
target_compile_options(mcts-halo-coverage PRIVATE -Xcompiler=-no-pie)
set_standards(mcts-halo-coverage)
set_options(mcts-halo-coverage)
target_link_options(mcts-halo-coverage PUBLIC -rdynamic)

add_executable(mcts-halo-anticorr halo_anticorr.cu ops_halo_exchange.cu)
target_link_libraries(mcts-halo-anticorr sched)
target_compile_options(mcts-halo-anticorr PRIVATE -Xcompiler=-fno-pie)
target_compile_options(mcts-halo-anticorr PRIVATE -Xcompiler=-no-pie)
set_standards(mcts-halo-anticorr)
set_options(mcts-halo-anticorr)
target_link_options(mcts-halo-anticorr PUBLIC -rdynamic)

add_executable(mcts-halo-balance halo_balance.cu ops_halo_exchange.cu)
target_link_libraries(mcts-halo-balance sched)
target_compile_options(mcts-halo-balance PRIVATE -Xcompiler=-fno-pie)
target_compile_options(mcts-halo-balance PRIVATE -Xcompiler=-no-pie)
set_standards(mcts-halo-balance)
set_options(mcts-halo-balance)
target_link_options(mcts-halo-balance PUBLIC -rdynamic)