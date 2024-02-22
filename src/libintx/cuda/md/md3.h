#ifndef LIBINTX_CUDA_MD_MD3_H
#define LIBINTX_CUDA_MD_MD3_H

#include "libintx/cuda/forward.h"
#include "libintx/engine.h"
#include "libintx/tensor.h"

namespace libintx::cuda::md {

  struct Basis1;
  struct Basis2;

  struct ERI3 : IntegralEngine<3,2> {

    ERI3(
      const Basis<Gaussian> &bra,
      const Basis<Gaussian> &ket,
      cudaStream_t stream
    );

    ~ERI3();

    void compute(
      const std::vector<int> &bra,
      const std::vector<Index2> &ket,
      double*,
      std::array<size_t,2>
    ) override;

  private:

    template<int>
    double* buffer(size_t);

    template<int Bra, int Ket>
    void compute(const Basis1&, const Basis2&, TensorRef<double,2>, cudaStream_t);

    template<int,int,int>
    auto compute_v2(
      const Basis1& x,
      const Basis2& ket,
      TensorRef<double,2> XCD,
      cudaStream_t stream
    );

  private:

    Basis<Gaussian> bra_, ket_;
    cudaStream_t stream_;
    struct Memory;
    std::unique_ptr<Memory> memory_;

  };

}

#endif /* LIBINTX_CUDA_MD_MD3_H */
