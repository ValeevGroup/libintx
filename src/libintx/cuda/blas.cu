#include "libintx/cuda/blas.h"

//#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm_batched.h"

namespace libintx::cuda {

  template<int Tile>
  __global__
  void transpose(size_t M, size_t N, const double *A, size_t ldA, double *T, size_t ldT) {
    assert(M <= ldA);
    assert(N <= ldT);
    __shared__ double t[Tile][Tile+1];
    for (int y = threadIdx.y; y < Tile; y += blockDim.y) {
      int i = threadIdx.x + blockIdx.x*Tile;
      int j = y + blockIdx.y*Tile;
      if (i >= M || j >= N) break;
      t[threadIdx.x][y] = A[i + j*ldA];
      //printf("A[%i,%i]=%f\n", i, j, t[threadIdx.x][y]);
    }
    __syncthreads();
    for (int y = threadIdx.y; y < Tile; y += blockDim.y) {
      int i = y + blockIdx.x*Tile;
      int j = threadIdx.x + blockIdx.y*Tile;
      //printf("A'[%i,%i]=%f\n", j, i, t[y][threadIdx.x]);
      if (i >= M || j >= N) break;
      T[j + i*ldT] = t[y][threadIdx.x];
      //printf("t[%i,%i]=%f\n", y, threadIdx.x, t[y][threadIdx.x]);
      //T[threadIdx.x + blockIdx.y*32 + (i+blockIdx.x*32)*N] = t[i][threadIdx.x];
      //T[(i+blockIdx.x*32)*N] = t[i][threadIdx.x];
    }
  }

  void transpose(
    size_t M, size_t N,
    const double *A, size_t ldA,
    double *T, size_t ldT,
    cudaStream_t stream)
  {
    if (!M || !N) return;
    constexpr int DimX = 16;
    constexpr size_t batches = 1;
    dim3 g = { (M+DimX-1)/DimX, (N+DimX-1)/DimX, batches };
    dim3 b = { DimX, 4 };
    transpose<DimX><<<g,b,0,stream>>>(M, N, A, ldA, T, ldT);
  }

  template<int Tile>
  __global__
  void batch_transpose(
    size_t M, size_t N,
    const double *A, size_t ldA,
    double *T, size_t ldT)
  {
    assert(M <= ldA);
    assert(N <= ldT);
    __shared__ double t[Tile][Tile+1];
    for (int y = threadIdx.y; y < Tile; y += blockDim.y) {
      int i = threadIdx.x + blockIdx.x*Tile;
      int j = y + blockIdx.y*Tile;
      if (i >= M || j >= N) break;
      t[threadIdx.x][y] = A[i + j*ldA + blockIdx.z*ldA*N];
      //printf("A[%i,%i]=%f\n", i, j, t[threadIdx.x][y]);
    }
    __syncthreads();
    for (int y = threadIdx.y; y < Tile; y += blockDim.y) {
      int i = y + blockIdx.x*Tile;
      int j = threadIdx.x + blockIdx.y*Tile;
      //printf("A'[%i,%i]=%f\n", j, i, t[y][threadIdx.x]);
      if (i >= M || j >= N) break;
      T[j + i*ldT + blockIdx.z*ldT*M] = t[y][threadIdx.x];
      //printf("t[%i,%i]=%f\n", y, threadIdx.x, t[y][threadIdx.x]);
      //T[threadIdx.x + blockIdx.y*32 + (i+blockIdx.x*32)*N] = t[i][threadIdx.x];
      //T[(i+blockIdx.x*32)*N] = t[i][threadIdx.x];
    }
  }

  // S[M,N,batches] -> T[N,M,batches]
  void batch_transpose(
    size_t M, size_t N,
    const double *A, size_t ldA,
    double *T, size_t ldT,
    size_t batches,
    cudaStream_t stream)
  {
    if (!M || !N) return;
    constexpr int Tile = 16;
    dim3 g = { (M+Tile-1)/Tile, (N+Tile-1)/Tile, batches };
    dim3 b = { Tile, 4 };
    batch_transpose<Tile><<<g,b,0,stream>>>(M, N, A, ldA, T, ldT);
  }


  template<typename LayoutA, typename LayoutB, typename LayoutC>
  void batch_gemm(
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
        TensorRef<const double,LayoutA>{A, ldA}, strideA,
        TensorRef<const double,LayoutB>{B, ldB}, strideB,
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
  void batch_gemm<ColumnMajor,ColumnMajor,ColumnMajor>(
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
  void batch_gemm<ColumnMajor,RowMajor,ColumnMajor>(
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
  void batch_gemm<RowMajor,RowMajor,ColumnMajor>(
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
