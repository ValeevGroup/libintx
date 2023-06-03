#include "libintx/cuda/blas.h"

//#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm_batched.h"

namespace libintx::cuda {

  template<typename LayoutA, typename LayoutB, typename LayoutC>
  void GemmBatched(
    int M, int N, int K,
    double alpha,
    const double *A, int64_t ldA, int64_t strideA,
    const double *B, int64_t ldB, int64_t strideB,
    double beta,
    double *C, int64_t ldC, int64_t strideC,
    int batches,
    cudaStream_t stream)
  {

    using namespace cutlass::gemm;
    using cutlass::TensorRef;
    using cutlass::gemm::GemmShape;

    using Gemm = cutlass::gemm::device::GemmBatched<
      double, LayoutA,
      double, LayoutB,
      double, LayoutC,
      double,
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm70,
      GemmShape<64, 32, 8>,
      GemmShape<64, 8, 8>
      >;

    Gemm gemm_op;

    cutlass::Status status = gemm_op(
      typename Gemm::Arguments{
        GemmCoord{M, N, K},
        //TensorRef<const double,RowMajor>{PX, K*bra.N}, K*N,
        TensorRef<const double,RowMajor>{A, ldA}, strideA,
        TensorRef<const double,RowMajor>{B, ldB}, strideB,
        TensorRef<const double,LayoutC>{C, ldC}, strideC,
        TensorRef<double, LayoutC>{C, ldC}, strideC,
        typename Gemm::EpilogueOutputOp::Params{alpha, beta},
        (int)batches
      },
      stream
    );

    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("cuda::gemm error");
    }

  }

  template
  void GemmBatched<RowMajor,RowMajor,RowMajor>(
    int M, int N, int K,
    double alpha,
    const double *A, int64_t ldA, int64_t strideA,
    const double *B, int64_t ldB, int64_t strideB,
    double beta,
    double *C, int64_t ldC, int64_t strideC,
    int batches,
    cudaStream_t stream
  );

  template
  void GemmBatched<RowMajor,RowMajor,ColumnMajor>(
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
