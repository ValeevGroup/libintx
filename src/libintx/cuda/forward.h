#ifndef LIBINTX_CUDA_FORWARD_H
#define LIBINTX_CUDA_FORWARD_H

#include "libintx/forward.h"

#include <memory>
#include <vector>

typedef struct CUstream_st* cudaStream_t;

namespace boys::cuda {

  template<int Order, int M, int MaxT, int Segments>
  struct Chebyshev;

}

namespace libintx::cuda {

  using Boys = boys::cuda::Chebyshev<7,40,117,117*7>;

  const Boys& boys();

  std::shared_ptr<const Boys> boys(int device);

}

#endif /* LIBINTX_CUDA_FORWARD_H */
