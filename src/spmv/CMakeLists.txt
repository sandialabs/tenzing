# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
# terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
# software.

add_library(spmv
ops_spmv.cu
)

set_standards(spmv)
set_options(spmv)
target_link_libraries(spmv sched)

