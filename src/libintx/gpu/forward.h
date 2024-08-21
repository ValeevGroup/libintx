#ifndef LIBINTX_CUDA_FORWARD_H
#define LIBINTX_CUDA_FORWARD_H

#include "libintx/forward.h"

#include <memory>
#include <vector>

struct CUstream_st;

namespace libintx {

  typedef CUstream_st* gpuStream_t;

}

namespace boys::gpu {

  template<int Order, int M, int MaxT, int Segments>
  struct Chebyshev;

}

namespace libintx::gpu {

  using Boys = boys::gpu::Chebyshev<7,40,117,117*7>;

  const Boys& boys();

  std::shared_ptr<const Boys> boys(int device);

}

#endif /* LIBINTX_CUDA_FORWARD_H */
