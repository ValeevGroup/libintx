#ifndef LIBINTX_CUDA_BLAS_H
#define LIBINTX_CUDA_BLAS_H

#include "libintx/cuda/api/api.h"
//#include "cublas_v2.h"
#include "cutlass/layout/layout.h"

namespace libintx::cuda {

  using cutlass::layout::RowMajor;
  using cutlass::layout::ColumnMajor;

  template<typename LayoutA, typename LayoutB, typename LayoutC>
  void GemmBatched(
    int M, int N, int K,
    double alpha,
    const double *A, int64_t ldA, int64_t strideA,
    const double *B, int64_t ldB, int64_t strideB,
    double beta,
    double *C, int64_t ldC, int64_t strideC,
    int batches,
    cudaStream_t stream
  );

}

#endif /* LIBINTX_CUDA_BLAS_H */
