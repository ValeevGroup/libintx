#ifndef LIBINTX_CUDA_MD_HERMITIAN_H
#define LIBINTX_CUDA_MD_HERMITIAN_H

#include "libintx/shell.h"
#include "libintx/cuda/api/api.h"
#include "libintx/config.h"

namespace libintx::cuda::md {

  struct alignas(8) Hermitian {
    Double<3> r;
    double exp;
    double C;
  };

  struct Basis1 {
    const int L, K;
    device::vector<Hermitian> H;
  };

  struct Basis2 {
    const std::pair<int,int> L;
    const int K, N;
    device::vector<double> H;
  };

  Basis2 make_basis(
    const Basis<Gaussian> &A
    const Basis<Gaussian> &B,
    const std::vector< std::pair<int,int> > &pairs,
    device::vector<double> &data
  );

}


#endif /* LIBINTX_CUDA_MD_HERMITIAN_H */
