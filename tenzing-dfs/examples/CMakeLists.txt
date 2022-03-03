# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
# terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
# software.

add_executable(spmv spmv.cu)
target_link_libraries(spmv tenzing-dfs)
tenzing_set_standards(spmv)