#ifndef LIBINTX_CUDA_MD_ENGINE_H
#define LIBINTX_CUDA_MD_ENGINE_H

#include "libintx/cuda/eri.h"
#include "libintx/cuda/api/api.h"
#include <memory>

namespace libintx::cuda::md {

  template<int Order, typename ... Args>
  std::unique_ptr< IntegralEngine<Order,2> > eri(const Args& ...) = delete;

  template<>
  std::unique_ptr< IntegralEngine<4,2> > eri(
    const Basis<Gaussian>&,
    const Basis<Gaussian>&,
    const cudaStream_t&
  );

}

#endif /* LIBINTX_CUDA_MD_ENGINE_H */
