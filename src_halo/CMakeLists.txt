add_executable(halo halo.cu ops_halo_exchange.cu)
target_link_libraries(halo sched)
set_standards(halo)
set_options(halo)