#ifndef LIBINTX_GPU_BLAS_H
#define LIBINTX_GPU_BLAS_H

#include "libintx/gpu/forward.h"

namespace libintx::gpu {

  enum Order {
    RowMajor, ColumnMajor
  };

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

  template<Order LayoutA, Order LayoutB, Order LayoutC>
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
