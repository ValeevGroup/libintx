#ifndef LIBINTX_CUDA_MD_ENGINE_H
#define LIBINTX_CUDA_MD_ENGINE_H

#include "libintx/engine.h"
#include "libintx/shell.h"
#include "libintx/gpu/forward.h"

#include <memory>

namespace libintx::gpu::md {

  template<int Bra, int Ket>
  std::unique_ptr< IntegralEngine<Bra,Ket> > integral_engine(
    const Basis<Gaussian>& bra,
    const Basis<Gaussian>& ket,
    const gpuStream_t& stream
  ) = delete;

  template<>
  std::unique_ptr< IntegralEngine<1,2> > integral_engine(
    const Basis<Gaussian>& bra,
    const Basis<Gaussian>& ket,
    const gpuStream_t& stream
  );

  template<>
  std::unique_ptr< IntegralEngine<2,2> > integral_engine(
    const Basis<Gaussian>& bra,
    const Basis<Gaussian>& ket,
    const gpuStream_t& stream
  );

  template<int Order, typename ... Args>
  auto eri(const Args& ... args) {
    constexpr int Bra = Order/2;
    constexpr int Ket = (Order+1)/2;
    return integral_engine<Bra,Ket>(args...);
  }

}

#endif /* LIBINTX_CUDA_MD_ENGINE_H */
