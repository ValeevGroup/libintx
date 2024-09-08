#ifndef LIBINTX_GPU_MD_ENGINE_H
#define LIBINTX_GPU_MD_ENGINE_H

#include "libintx/gpu/forward.h"
#include "libintx/gpu/engine.h"
#include "libintx/shell.h"
#include "libintx/tensor.h"
#include <memory>

namespace libintx::gpu::md {

  struct Basis1;
  struct Basis2;

  template<int N>
  struct IntegralEngine;

  template<>
  struct IntegralEngine<3> : gpu::IntegralEngine<3> {

    IntegralEngine(
      const Basis<Gaussian> &bra,
      const Basis<Gaussian> &ket,
      gpuStream_t stream
    );

    ~IntegralEngine();

    void compute(
      Operator op,
      const std::vector<Index1> &bra,
      const std::vector<Index2> &ket,
      double*,
      const std::array<size_t,2>&
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


  template<>
  struct IntegralEngine<4> : gpu::IntegralEngine<4> {

    IntegralEngine(
      const Basis<Gaussian> &bra,
      const Basis<Gaussian> &ket,
      gpuStream_t stream
    );

    ~IntegralEngine();

    void compute(
      Operator,
      const std::vector<Index2> &bra,
      const std::vector<Index2> &ket,
      double*,
      const std::array<size_t,2>&
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


} // libintx::gpu::md

#endif /* LIBINTX_GPU_MD_ENGINE_H */
