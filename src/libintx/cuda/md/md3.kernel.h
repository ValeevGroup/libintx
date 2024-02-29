#include "libintx/cuda/forward.h"
#include "libintx/cuda/md/basis.h"
#include "libintx/cuda/md/md.kernel.h"
#include "libintx/engine/md/r1.h"
#include "libintx/engine/md/r1/recurrence.h"
#include "libintx/engine/md/hermite.h"
#include "libintx/cuda/blas.h"

#include "libintx/cuda/api/thread_group.h"
#include "libintx/boys/cuda/chebyshev.h"

#include "libintx/config.h"
#include "libintx/math.h"
#include "libintx/utility.h"

namespace libintx::cuda::md::kernel {

  namespace cart = libintx::cartesian;
  namespace herm = libintx::hermite;

  using libintx::pure::cartesian_to_pure;

  // compute [q,ij,x,kl]
  template<typename ThreadBlock, int MinBlocks, int X, int Ket, typename Boys>
  __global__
  __launch_bounds__(ThreadBlock::size(),MinBlocks)
  static void compute_q_x(
    const Basis1<X> bra,
    const Basis2<Ket> ket,
    const std::pair<int,int> K,
    const Boys boys,
    auto QX)
  {

    static constexpr int L = bra.L+ket.L;
    static constexpr int NP = bra.nherm;
    static constexpr int NQ = ket.nherm;

    constexpr ThreadBlock thread_block;
    constexpr int num_threads = ThreadBlock::size();
    int rank = thread_block.thread_rank();

    struct Shmem {
      Hermite x,cd;
      double R[nherm2(L)];
      array<double,3> PQ;
      double inv_2p;
    };
    __shared__ Shmem shmem;

    const auto &ij = blockIdx.x;
    const auto &kl = blockIdx.y;

    memcpy1(bra.hdata(ij,K.first), &shmem.x, thread_block);
    memcpy1(ket.hdata(kl,K.second), &shmem.cd, thread_block);
    thread_block.sync();

    auto &P = shmem.x.r;
    auto &Q = shmem.cd.r;
    auto &PQ = shmem.PQ;
    auto &R = shmem.R;

    if (rank < 3) PQ[rank] = P[rank] - Q[rank];

    thread_block.sync();

    if (rank <= L) {

      auto &x = shmem.x;
      auto &cd = shmem.cd;

      double p = x.exp;
      double q = cd.exp;
      double C = x.C*cd.C;
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

    for (int iq = threadIdx.x; iq < NQ; iq += thread_block.x) {

      const auto q = kernel::orbitals(ket)[iq];
      int phase = (q.L()%2 == 0 ? +1 : -1);

      double r[NP] = {};
      double v[bra.nbf] = {};

      if (K.first) {
        for (int ix = 0; ix < bra.nbf; ++ix) {
          v[ix] = QX(iq, ij, ix, kl);
        }
      }

      for (int ip = 0; ip < NP; ++ip) {
        const auto &p = kernel::orbitals(bra)[ip];
        r[ip] = R[herm::index2(p+q)];
      }

      foreach(
        std::make_index_sequence<ncart(X)>(),
        [&](auto ix) {
          constexpr auto x = std::get<ix.value>(cart::shell<X>());
          auto h = [&](auto&& ... p) {
            constexpr int idx = herm::index1(p.value...);
            return r[idx];
          };
          r[herm::index1(x)] = hermite_to_cartesian<x[0],x[1],x[2]>(h, shmem.x.inv_2_exp);
        }
      );

      cartesian_to_pure<X>(
        [&](auto x, auto u) {
          QX(iq, ij, index(x), kl) = phase*u + v[index(x)];
        },
        [&](auto x) {
          return r[herm::index1(x)];
        }
      );

    }

  }


}
