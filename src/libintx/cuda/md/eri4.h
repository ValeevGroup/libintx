#ifndef LIBINTX_CUDA_MD_ERI4_H
#define LIBINTX_CUDA_MD_ERI4_H

#include "libintx/cuda/md/engine.h"
#include "libintx/cuda/md/basis.h"
#include "libintx/cuda/api/api.h"

namespace libintx::cuda::md {

  struct ERI4 : IntegralEngine<4> {

    ERI4(
      const Basis<Gaussian> &basis,
      const Basis<Gaussian> &df_basis,
      cudaStream_t stream
    );

    void compute(
      const std::vector<Index2> &bra,
      const std::vector<Index2> &ket,
      double *data
    ) override;

  private:

    template<int AB, int X>
    static void compute(ERI4 &eri, const Basis2&, const Basis2&, double*, size_t, cudaStream_t);

    template<int A, int B, int X>
    static void compute(ERI4 &eri, const Basis2&, const Basis2&, double*, size_t, cudaStream_t);

  private:

    // //cublasHandle_t handle_;
    Basis<Gaussian> bra_, ket_;
    device::vector<double> p_, q_;
    device::vector<double> pq_, abq_;
    cudaStream_t stream_;

  };

}


#endif /* LIBINTX_CUDA_MD_ERI4_H */
