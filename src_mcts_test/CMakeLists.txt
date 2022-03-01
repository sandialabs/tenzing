# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
# terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
# software.


add_executable(mcts mcts.cpp)
target_link_libraries(mcts sched)
target_compile_options(mcts PRIVATE -fno-pie)
target_compile_options(mcts PRIVATE -no-pie)
tenzing_set_standards(mcts)
tenzing_set_options(mcts)

# function names in back trace
target_link_options(mcts PUBLIC -rdynamic)

add_executable(pingpong pingpong.cpp)
target_link_libraries(pingpong sched)
target_compile_options(pingpong PRIVATE -fno-pie)
target_compile_options(pingpong PRIVATE -no-pie)
tenzing_set_standards(pingpong)
tenzing_set_options(pingpong)
target_link_options(pingpong PUBLIC -rdynamic)