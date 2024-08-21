#include "libintx/gpu/forward.h"
#include "libintx/boys/gpu/chebyshev.h"
#include "libintx/gpu/api/api.h"

#include <mutex>
#include <map>

namespace libintx::gpu {

  const Boys& boys() {
    return *boys(gpu::current_device::get());
  }

  std::shared_ptr<const Boys> boys(int device) {
    static std::mutex mutex;
    static std::map<int, std::shared_ptr<const Boys> > boys;
    std::unique_lock<std::mutex> lock(mutex);
    auto &b = boys[device];
    if (!b) {
      b = std::make_unique<Boys>(device);
    }
    return b;
  }

}
