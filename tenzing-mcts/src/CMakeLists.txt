add_library(tenzing-mcts
counters.cpp
mcts_node.cpp
mcts.cpp
)
target_include_directories(tenzing-mcts PUBLIC ${tenzing_SOURCE_DIR}/tenzing-mcts/include)
target_link_libraries(tenzing-mcts tenzing)
tenzing_set_standards(tenzing-mcts)