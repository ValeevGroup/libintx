#ifndef LIBINTX_GPU_MD_ERI4_H
#define LIBINTX_GPU_MD_ERI4_H

#include "libintx/gpu/md/engine.h"
#include "libintx/tensor.h"

namespace libintx::gpu::md {

  struct Basis2;

  struct ERI4 : IntegralEngine<2,2> {

    ERI4(
      const Basis<Gaussian> &bra,
      const Basis<Gaussian> &ket,
      gpuStream_t stream
    );

    ~ERI4();

    void compute(
      const std::vector<Index2> &bra,
      const std::vector<Index2> &ket,
      double*,
      std::array<size_t,2>
    ) override;

  private:

    template<int Bra, int Ket>
    void compute(const Basis2&, const Basis2&, TensorRef<double,2>, gpuStream_t);

    template<int,int,int,int>
    auto compute_v0(
      const Basis2& bra,
      const Basis2& ket,
      TensorRef<double,2> ABCD,
      gpuStream_t stream
    );

    template<int,int,int,int>
    auto compute_v1(
      const Basis2& bra,
      const Basis2& ket,
      TensorRef<double,2> ABCD,
      gpuStream_t stream
    );

    template<int,int,int,int>
    auto compute_v2(
      const Basis2& bra,
      const Basis2& ket,
      TensorRef<double,2> ABCD,
      gpuStream_t stream
    );

    template<int>
    double* allocate(size_t);

  private:
    Basis<Gaussian> bra_, ket_;
    gpuStream_t stream_;
    struct Memory;
    std::unique_ptr<Memory> memory_;

  };

}

#endif /* LIBINTX_GPU_MD_ERI4_H */
