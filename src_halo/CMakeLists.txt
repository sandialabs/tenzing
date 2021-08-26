add_executable(halo halo.cu ops_halo_exchange.cu)
target_link_libraries(halo sched)
target_compile_options(halo PRIVATE -Xcompiler=-fno-pie)
target_compile_options(halo PRIVATE -Xcompiler=-no-pie)
set_standards(halo)
set_options(halo)

# function names in back trace
target_link_options(halo PUBLIC -rdynamic)