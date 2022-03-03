# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
# terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
# software.

add_library(tenzing-dfs dfs.cpp)
target_include_directories(tenzing-dfs PUBLIC ${tenzing_SOURCE_DIR}/tenzing-dfs/include)
target_link_libraries(tenzing-dfs tenzing)
tenzing_set_standards(tenzing-dfs)