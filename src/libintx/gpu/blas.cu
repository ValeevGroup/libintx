#include "libintx/gpu/blas.h"
#include "libintx/gpu/api/runtime.h"

#ifdef __CUDACC__
//#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm_batched.h"
#endif

namespace libintx::gpu {

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
    gpuStream_t stream)
  {
    if (!M || !N) return;
    constexpr uint DimX = 16;
    constexpr size_t batches = 1;
    dim3 g = { uint((M+DimX-1)/DimX), uint((N+DimX-1)/DimX), batches };
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
    gpuStream_t stream)
  {
    if (!M || !N) return;
    constexpr int Tile = 16;
    dim3 g = { uint((M+Tile-1)/Tile), uint((N+Tile-1)/Tile), uint(batches) };
    dim3 b = { Tile, 4 };
    batch_transpose<Tile><<<g,b,0,stream>>>(M, N, A, ldA, T, ldT);
  }

#ifdef __CUDACC__

  template<Order order>
  auto layout() {
    if constexpr (order == RowMajor) return cutlass::layout::RowMajor{};
    if constexpr (order == ColumnMajor) return cutlass::layout::ColumnMajor{};
  };

  template<Order order>
  using layout_t = decltype(layout<order>());

#endif

  template<Order LayoutA, Order LayoutB, Order LayoutC>
  void batch_gemm(
    int M, int N, int K,
    double alpha,
    const double *A, int64_t ldA, int64_t strideA,
    const double *B, int64_t ldB, int64_t strideB,
    double beta,
    double *C, int64_t ldC, int64_t strideC,
    int batches,
    gpuStream_t stream)
  {

#ifdef __CUDACC__

    using namespace cutlass::gemm;
    using cutlass::TensorRef;
    using cutlass::gemm::GemmShape;

    using Gemm = cutlass::gemm::device::GemmBatched<
      double, layout_t<LayoutA>,
      double, layout_t<LayoutB>,
      double, layout_t<LayoutC>,
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
        TensorRef<const double,layout_t<LayoutA>>{A, ldA}, strideA,
        TensorRef<const double,layout_t<LayoutB>>{B, ldB}, strideB,
        TensorRef<const double,layout_t<LayoutC>>{C, ldC}, strideC,
        TensorRef<double, layout_t<LayoutC>>{C, ldC}, strideC,
        typename Gemm::EpilogueOutputOp::Params{alpha, beta},
        (int)batches
      },
      stream
    );

    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("cuda::gemm error");
    }

#endif

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
    gpuStream_t stream
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
    gpuStream_t stream
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
    gpuStream_t stream
  );


}
