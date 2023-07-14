#include "libintx/cuda/forward.h"
#include "libintx/cuda/md/engine.h"
#include "libintx/cuda/md/basis.h"
#include "libintx/engine/md/r1.h"
#include "libintx/engine/md/r1/recurrence.h"
#include "libintx/cuda/blas.h"

#include "libintx/cuda/api/kernel.h"
#include "libintx/cuda/api/thread_group.h"
#include "libintx/boys/cuda/chebyshev.h"

#include "libintx/config.h"
#include "libintx/math.h"
#include "libintx/utility.h"

namespace libintx::cuda::md::eri4 {

  namespace cart = libintx::cartesian;

  LIBINTX_GPU_DEVICE
  constexpr const auto index2_lookup_table = make_array<int,4*LMAX+1>(&math::figurate<3,int>);

  LIBINTX_GPU_DEVICE
  constexpr auto orbitals = hermitian::orbitals<2*LMAX>;

  template<int ... Args>
  struct Basis2;

  template<int AB>
  struct Basis2<AB> {
    static constexpr int L = AB;
    static constexpr int nherm = nherm2(L);
    const int nbf;
    const int K;
    const int N;
    const double *data;
    const int stride;
    Basis2(Shell a, Shell b, int K, int N, const double *H)
      : nbf(libintx::nbf(a)*libintx::nbf(b)),
        K(K), N(N), data(H),
        stride(sizeof(Hermitian)/sizeof(double)+nherm*nbf)
    {
    }
    LIBINTX_GPU_ENABLED
    auto hdata(int p, int k) const {
      return reinterpret_cast<const Hermitian*>(data + k*stride + p*K*stride);
    }
    LIBINTX_GPU_ENABLED
    auto gdata(int p, int k) const {
      return reinterpret_cast<const double*>(hdata(p,k)+1);
    }
  };

  template<int _A, int _B>
  struct Basis2<_A,_B> {
    static constexpr int L = _A + _B;
    static constexpr int nherm = nherm2(L);
    static constexpr int nbf = npure(_A)*npure(_B);
    static constexpr int stride = sizeof(Hermitian)/sizeof(double) + nherm*nbf;
    const int K;
    const int N;
    const double *data;
    static constexpr int np = 3+3+nherm*nbf;
    Basis2(int K, int N, const double *H)
      : K(K), N(N), data(H)
    {
    }
    LIBINTX_GPU_ENABLED
    auto hdata(int p, int k = 0) const {
      return reinterpret_cast<const Hermitian*>(data + k*stride + p*K*stride);
    }
    LIBINTX_GPU_ENABLED
    auto gdata(int p, int k = 0) const {
      return reinterpret_cast<const double*>(hdata(p,k)+1);
    }
  };


  template<typename ThreadBlock, int MinBlocks, int Bra, int Ket, typename Boys>
  __global__
  __launch_bounds__(ThreadBlock::size(),MinBlocks)
    static void eri4_pq(
      const Basis2<Bra> bra,
      const Basis2<Ket> ket,
      std::pair<int,int> k,
      const Boys boys,
      double *H)
  {

    using cartesian::orbital;
    using cartesian::index;

    static constexpr int L = bra.L+ket.L;
    static constexpr int NP = bra.nherm;
    static constexpr int NQ = ket.nherm;
    static constexpr int NPQ = NP*NQ;

    constexpr ThreadBlock thread_block;
    constexpr int num_threads = ThreadBlock::size();
    int rank = thread_block.thread_rank();

    __shared__ Hermitian ab,cd;
    __shared__ double R[nherm2(L)];

    memcpy1(bra.hdata(blockIdx.x,k.first), &ab, thread_block);
    memcpy1(ket.hdata(blockIdx.y,k.second), &cd, thread_block);
    thread_block.sync();

    auto &Q = cd.r;
    auto &P = ab.r;

    __shared__ array<double,3> PQ;
    if (rank < 3) PQ[rank] = P[rank] - Q[rank];

    thread_block.sync();

    if (rank <= L) {

      double p = ab.exp;
      double q = cd.exp;
      double C = ab.C*cd.C;
      //C *= 2*std::pow(M_PI,2.5);

      double alpha = (p*q)/(p+q);
      double T = alpha*norm(P,Q);
      double Fm = boys.compute(T, rank);
      double pq = p*q;
      C *= rsqrt(pq*pq*(p+q));
      //double Kab = exp(-(a*b)/p*norm(P));
      //double Kcd = 1;//exp(-norm(Q));
      //C *= Kcd;
      for (int i = 0; i <= L; ++i) {
        if (i == rank) break;
        C *= -2*alpha;
      }
      R[rank] = C*Fm;
      //printf("ip=%i vc=%f\n", rank, R[rank]);
      //printf("T=%f (0)=%f\n", T, R[rank]);
    }
    thread_block.sync();

    if constexpr (L > 0) {
      namespace r1 = libintx::md::r1;
      r1::compute<L>(r1::recurrence, PQ, R, thread_block);
      thread_block.sync();
    }

    // N.B blockIdx order is [p,q]
    int xy = (
      (blockIdx.x*gridDim.y + blockIdx.y)*NPQ +
      threadIdx.x + threadIdx.y*NP
    );

    for (int iq = 0; iq < NQ; iq += thread_block.y) {
      if (threadIdx.y + iq >= NQ) break;

      const auto q = eri4::orbitals[threadIdx.y + iq];
      const auto* __restrict__ index2 = index2_lookup_table.data + q.L();
      int phase = (q.L()%2 == 0 ? +1 : -1);

      for (int ip = 0; ip < NP; ip += thread_block.x) {
        if (threadIdx.x + ip >= NP) break;
        const auto p = eri4::orbitals[threadIdx.x + ip];
        auto ipq = index(p+q) + index2[p.L()];
        H[ip + iq*NP + xy] = phase*R[ipq];
      }

    }

  }


  template<int Side, int Bra>
  void eri4_ap_px(
    Basis2<Bra> bra, int NX,
    const double *H,
    const double *PX,
    double alpha, double beta,
    double *AX,
    cudaStream_t stream)
  {

    //int M = ket.nherm*ket.N;
    int N = bra.nbf;
    int K = bra.nherm;
    int batches = bra.N;

    using Layout = std::conditional_t<Side, RowMajor, ColumnMajor>;

    //printf("gemm NX=%i, N=%i, K=%i\n", NX, N, K);

    auto *A = PX;
    int ldA = (Side == 0 ? K : K*batches);
    int strideA = (Side == 0 ? K*NX : K);

    auto *B = H;
    int strideB = bra.K*bra.stride;
    int ldB = N;

    auto *C = AX;
    int ldC = (Side == 0 ? NX : batches*N);
    int strideC = (Side == 0 ? NX*N : N);

    GemmBatched<RowMajor, RowMajor, Layout>(
      NX, N, K,
      alpha,
      A, ldA, strideA,
      B, ldB, strideB,
      beta,
      C, ldC, strideC,
      bra.N,
      stream
    );

  }

}
