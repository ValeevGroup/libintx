#ifndef LIBINTX_CUDA_MD_BASIS_H
#define LIBINTX_CUDA_MD_BASIS_H

#include "libintx/shell.h"
#include "libintx/gpu/forward.h"
#include "libintx/gpu/api/api.h"

namespace libintx::cuda::md {

  struct alignas(8) Hermite {
    double exp;
    double C;
    Double<3> r;
    double inv_2_exp;

    LIBINTX_GPU_ENABLED
    static auto* hdata(double *p) {
      return reinterpret_cast<Hermite*>(p);
    }

    LIBINTX_GPU_ENABLED
    static auto* hdata(const double *p) {
      return reinterpret_cast<const Hermite*>(p);
    }

    LIBINTX_GPU_ENABLED
    static auto* gdata(double *p) {
      return reinterpret_cast<double*>(hdata(p)+1);
    }

    LIBINTX_GPU_ENABLED
    static auto* gdata(const double *p) {
      return reinterpret_cast<const double*>(hdata(p)+1);
    }

    LIBINTX_GPU_ENABLED
    static constexpr size_t extent(const Shell &A, const Shell &B) {
      return (sizeof(Hermite)/sizeof(double) + nbf(A)*nbf(B)*nherm2(A.L+B.L));
    }

  };

  struct Basis1 {
    const int L, K, N;
    const Hermite *data;
  };

  struct Basis2 {
    static constexpr size_t alignment = 128;
    const Shell first, second;
    const int N, K;
    const double *data;
    const size_t k_stride;
    const double *pure_transform;
  };

  Basis1 make_basis(
    const Basis<Gaussian> &A,
    const std::vector<Index1> &idx,
    device::vector<Hermite> &H,
    cudaStream_t
  );

  Basis2 make_basis(
    const Basis<Gaussian> &A,
    const Basis<Gaussian> &B,
    const std::vector<Index2> &pairs,
    device::vector<double> &H,
    cudaStream_t
  );

}


#endif /* LIBINTX_CUDA_MD_BASIS_H */
