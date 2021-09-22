add_executable(mcts mcts.cpp)
target_link_libraries(mcts sched)
target_compile_options(mcts PRIVATE -fno-pie)
target_compile_options(mcts PRIVATE -no-pie)
set_standards(mcts)
set_options(mcts)

# function names in back trace
target_link_options(mcts PUBLIC -rdynamic)