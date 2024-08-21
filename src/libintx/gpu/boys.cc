#include "libintx/gpu/forward.h"
#include "libintx/boys/gpu/chebyshev.h"
#include "cuda/api/current_device.hpp"

#include <mutex>
#include <map>

namespace libintx::gpu {

  const Boys& boys() {
    return *boys(::cuda::device::current::get_id());
  }

  std::shared_ptr<const Boys> boys(int device) {
    static std::mutex mutex;
    static std::map<int, std::shared_ptr<const Boys> > boys;
    std::unique_lock<std::mutex> lock(mutex);
    auto &b = boys[device];
    if (!b) {
      b = std::make_unique<Boys>(::cuda::device::get(device));
    }
    return b;
  }

}
