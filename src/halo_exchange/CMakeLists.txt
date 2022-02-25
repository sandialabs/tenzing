# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
# terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
# software.

add_library(halo-exchange
ops_halo_exchange.cu
)

set_standards(halo-exchange)
set_options(halo-exchange)
target_link_libraries(halo-exchange tenzing-object)

