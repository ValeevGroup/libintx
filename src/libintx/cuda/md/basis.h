#ifndef LIBINTX_CUDA_MD_BASIS_H
#define LIBINTX_CUDA_MD_BASIS_H

#include "libintx/shell.h"
#include "libintx/cuda/forward.h"
#include "libintx/cuda/api/api.h"

namespace libintx::cuda::md {

  struct alignas(8) Hermite {
    Double<3> r;
    double exp;
    double C;

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
    const int L, K;
    device::vector<Hermite> H;
  };

  inline size_t nbf(const Basis1 &v) {
    return npure(v.L)*v.H.size();
  }

  struct Basis2 {
    const Shell first, second;
    const int K, N;
    const double *data;
  };

  Basis2 make_basis(
    const Basis<Gaussian> &A,
    const Basis<Gaussian> &B,
    const std::vector<Index2> &pairs,
    device::vector<double> &H,
    cudaStream_t
  );

}


#endif /* LIBINTX_CUDA_MD_BASIS_H */
