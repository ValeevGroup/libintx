#ifndef LIBINTX_GPU_MD_MD3_H
#define LIBINTX_GPU_MD_MD3_H

#include "libintx/gpu/forward.h"
#include "libintx/engine.h"
#include "libintx/tensor.h"

namespace libintx::gpu::md {

  struct Basis1;
  struct Basis2;

  struct ERI3 : IntegralEngine<1,2> {

    ERI3(
      const Basis<Gaussian> &bra,
      const Basis<Gaussian> &ket,
      gpuStream_t stream
    );

    ~ERI3();

    void compute(
      const std::vector<Index1> &bra,
      const std::vector<Index2> &ket,
      double*,
      std::array<size_t,2>
    ) override;

  private:

    template<int>
    double* allocate(size_t);

    template<int Bra, int Ket>
    void compute(const Basis1&, const Basis2&, TensorRef<double,2>, gpuStream_t);

    template<int,int,int>
    auto compute_v0(
      const Basis1& x,
      const Basis2& ket,
      TensorRef<double,2> XCD,
      gpuStream_t stream
    );

    template<int,int,int>
    auto compute_v2(
      const Basis1& x,
      const Basis2& ket,
      TensorRef<double,2> XCD,
      gpuStream_t stream
    );

  private:

    Basis<Gaussian> bra_, ket_;
    gpuStream_t stream_;
    struct Memory;
    std::unique_ptr<Memory> memory_;

  };

}

#endif /* LIBINTX_GPU_MD_MD3_H */
