#include "sched/event_synchronizer.hpp"

bool EventSynchronizer::is_synced_gpu_then_cpu(const std::shared_ptr<BoundGpuOp> &a,
                                               const std::shared_ptr<CpuOp> & /*b*/,
                                               const Sequence<BoundOp> &path) {
  // find a
  auto ai = find(path, a);
  if (path.end() == ai) {
    THROW_RUNTIME("couldn't find a " << a->name() << " in path");
  }

  // check for any existing CER ... CES combo
  for (auto ceri = ai; ceri < path.end(); ++ceri) {
    if (auto cer = std::dynamic_pointer_cast<CudaEventRecord>(*ceri)) {
      if (cer->stream() == a->stream()) {
        for (auto cesi = ceri + 1; cesi < path.end(); ++cesi) {
          if (auto ces = std::dynamic_pointer_cast<CudaEventSync>(*cesi)) {
            if (ces->event() == cer->event()) {
              return true;
            }
          }
        }
      }
    }
  }
  return false;
}