/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#include "sched/numa.hpp"

#include <numa.h>

#include "sched/macro_at.hpp"

void bind_to_local_memory() {
#if SCHED_USE_NUMA == 1
  numa_set_strict(1);
  STDERR("set numa_set_strict(1)");
  numa_set_bind_policy(1);
  STDERR("set numa_set_bind_policy(1)");

  numa_exit_on_warn = 1;
  STDERR("set numa_exit_on_warn = 1");
  numa_exit_on_error = 1;
  STDERR("set numa_exit_on_error = 1");

  
  int preferred = numa_preferred();
  STDERR("numa_preferred() = " << preferred);

/*
numa_all_nodes_ptr points to a bitmask that is allocated by the
       library with bits representing all nodes on which the calling
       task may allocate memory.  This set may be up to all nodes on the
       system, or up to the nodes in the current cpuset.  The bitmask is
       allocated by a call to numa_allocate_nodemask() using size
       numa_max_possible_node().  The set of nodes to record is derived
       from /proc/self/status, field "Mems_allowed".  The user should
       not alter this bitmask.
*/
    for (int i = 0 ; i < numa_max_possible_node(); ++i) {
        if (numa_bitmask_isbitset(numa_all_nodes_ptr, i)) {
            STDERR("all_nodes_ptr = " << i);
        }
    }
  

/*
numa_get_membind() returns the mask of nodes from which memory
       can currently be allocated.  If the returned mask is equal to
       numa_all_nodes, then memory allocation is allowed from all nodes.
*/
    bitmask *membind = numa_get_membind();
    for (int i = 0 ; i < numa_max_possible_node(); ++i) {
        if (numa_bitmask_isbitset(membind, i)) {
            STDERR("membind = " << i);
        }
    }

/*
numa_set_membind() sets the memory allocation mask.  The task
       will only allocate memory from the nodes set in nodemask.
       Passing an empty nodemask or a nodemask that contains nodes other
       than those in the mask returned by numa_get_mems_allowed() will
       result in an error.
*/

    numa_allocate_nodemask();


/*
      numa_get_mems_allowed() returns the mask of nodes from which the
       process is allowed to allocate memory in it's current cpuset
       context.  Any nodes that are not included in the returned bitmask
       will be ignored in any of the following libnuma memory policy
       calls.

numa_all_cpus_ptr points to a bitmask that is allocated by the
       library with bits representing all cpus on which the calling task
       may execute.  This set may be up to all cpus on the system, or up
       to the cpus in the current cpuset.  The bitmask is allocated by a
       call to numa_allocate_cpumask() using size
       numa_num_possible_cpus().  The set of cpus to record is derived
       from /proc/self/status, field "Cpus_allowed".  The user should
       not alter this bitmask.


numa_all_nodes_ptr points to a bitmask that is allocated by the
       library with bits representing all nodes on which the calling
       task may allocate memory.  This set may be up to all nodes on the
       system, or up to the nodes in the current cpuset.  The bitmask is
       allocated by a call to numa_allocate_nodemask() using size
       numa_max_possible_node().  The set of nodes to record is derived
       from /proc/self/status, field "Mems_allowed".  The user should
       not alter this bitmask.


numa_num_configured_nodes() returns the number of memory nodes in
       the system. This count includes any nodes that are currently
       disabled. This count is derived from the node numbers in
       /sys/devices/system/node. (Depends on the kernel being configured
       with /sys (CONFIG_SYSFS)).





*/



#endif
}