#ifndef LIBINTX_GPU_BLAS_H
#define LIBINTX_GPU_BLAS_H

#include "libintx/gpu/api/api.h"
//#include "cublas_v2.h"
#include "cutlass/layout/layout.h"

namespace libintx::gpu {

  using cutlass::layout::RowMajor;
  using cutlass::layout::ColumnMajor;

  void transpose(
    size_t M, size_t N,
    const double *S, size_t ldS,
    double *T, size_t ldT,
    gpuStream_t stream
  );

  // S[M,N,batches] -> T[N,M,batches]
  void batch_transpose(
    size_t M, size_t N,
    const double *S, size_t ldS,
    double *T, size_t ldT,
    size_t batches,
    gpuStream_t stream
  );

  template<typename LayoutA, typename LayoutB, typename LayoutC>
  void batch_gemm(
    int M, int N, int K,
    double alpha,
    const double *A, int64_t ldA, int64_t strideA,
    const double *B, int64_t ldB, int64_t strideB,
    double beta,
    double *C, int64_t ldC, int64_t strideC,
    int batches,
    gpuStream_t stream
  );

}

#endif /* LIBINTX_GPU_BLAS_H */
