#include "libintx/cuda/forward.h"
#include "libintx/cuda/md/basis.h"
#include "libintx/engine/md/r1/recurrence.h"

#include "libintx/cuda/api/thread_group.h"

#include "libintx/math.h"

namespace libintx::cuda::md::kernel {

  namespace cart = libintx::cartesian;
  namespace herm = libintx::hermite;

  LIBINTX_GPU_DEVICE
  constexpr auto orbitals = hermite::orbitals<2*LMAX>;

  template<int ... Args>
  struct Basis2;

  template<int AB>
  struct Basis2<AB> {
    static constexpr int L = AB;
    static constexpr int nherm = nherm2(L);
    const Shell first, second;
    const int nbf;
    const int K;
    const int N;
    const double *data;
    const int stride;
    Basis2(Shell a, Shell b, int K, int N, const double *H)
      : first(a), second(b),
        nbf(libintx::nbf(a)*libintx::nbf(b)),
        K(K), N(N), data(H),
        stride(sizeof(Hermite)/sizeof(double)+nherm*nbf)
    {
    }
    LIBINTX_GPU_ENABLED
    auto hdata(int p, int k) const {
      return reinterpret_cast<const Hermite*>(data + k*stride + p*K*stride);
    }
    LIBINTX_GPU_ENABLED
    auto gdata(int p, int k) const {
      return reinterpret_cast<const double*>(hdata(p,k)+1);
    }
  };


  // compute [p,q,kl,ij]
  template<typename ThreadBlock, int MinBlocks, int Bra, int Ket, typename Boys>
  __global__
  __launch_bounds__(ThreadBlock::size(),MinBlocks)
  static void compute_p_q(
    const Basis2<Bra> bra,
    const Basis2<Ket> ket,
    std::pair<int,int> k,
    const Boys boys,
    auto H)
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

    __shared__ Hermite ab,cd;
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

    for (int iq = threadIdx.y; iq < NQ; iq += thread_block.y) {
      const auto q = kernel::orbitals[iq];
      int phase = (q.L()%2 == 0 ? +1 : -1);
      for (int ip = threadIdx.x; ip < NP; ip += thread_block.x) {
        const auto p = kernel::orbitals[ip];
        auto ipq = herm::index2(p+q);
        H(ip,iq,blockIdx.y,blockIdx.x) = phase*R[ipq];
      }
    }

  }

}
