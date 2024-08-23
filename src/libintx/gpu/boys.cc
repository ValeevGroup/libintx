#include "libintx/gpu/boys.h"
#include "libintx/gpu/api/api.h"

#include <mutex>
#include <map>

namespace libintx::gpu {

  const Boys& boys() {
    int device = gpu::current_device::get();
    static std::mutex mutex;
    static std::map<int, std::shared_ptr<const Boys> > boys;
    std::unique_lock<std::mutex> lock(mutex);
    auto &b = boys[device];
    if (!b) {
      b = std::make_unique<Boys>();
    }
    return *b;
  }

}
