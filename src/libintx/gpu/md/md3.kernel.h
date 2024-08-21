#include "libintx/gpu/forward.h"
#include "libintx/gpu/md/basis.h"
#include "libintx/gpu/md/md.kernel.h"
#include "libintx/engine/md/r1.h"
#include "libintx/engine/md/r1/recurrence.h"
#include "libintx/engine/md/hermite.h"
#include "libintx/gpu/blas.h"
#include "libintx/gpu/api/thread_group.h"

#include "libintx/config.h"
#include "libintx/math.h"
#include "libintx/utility.h"

namespace libintx::cuda::md::kernel {

  namespace cart = libintx::cartesian;
  namespace herm = libintx::hermite;

  using libintx::pure::cartesian_to_pure;

  template<
    typename Bra, typename Ket,
    int DimX, int DimY, int DimZ,
    int MaxShmem,
    int MinBlocks = 2
    >
  struct md3_x_cd_kernel;

  // (ij,x,cd,kl) kernel {ij->DimX}
  template<
    typename Bra, typename Ket,
    int DimX,
    int MaxShmem,
    int MinBlocks
    >
  struct md3_x_cd_kernel<Bra,Ket,DimX,1,1,MaxShmem,MinBlocks>
    : md_v0_kernel_base<Bra,Ket,DimX,1,MaxShmem,MinBlocks>
  {
    static_assert(Bra::Centers == 1);
  };

  // [ij,x,cd,kl) kernel {ij->DimX,p->DimY}
  template<
    typename Bra, typename Ket,
    int DimX, int DimY,
    int MaxShmem,
    int MinBlocks
    >
  struct md3_x_cd_kernel<Bra,Ket,DimX,DimY,1,MaxShmem,MinBlocks>
    : md_x_cd_kernel_base<1,Bra,Ket,DimX,DimY,1,MaxShmem,MinBlocks>
  {
    static_assert(Bra::Centers == 1);
  };

  // [ij,x,cd,kl) kernel {ij->DimX,cd->DimZ}
  template<
    typename Bra, typename Ket,
    int DimX, int DimZ,
    int MaxShmem,
    int MinBlocks
    >
  struct md3_x_cd_kernel<Bra,Ket,DimX,1,DimZ,MaxShmem,MinBlocks> {

    static_assert(Bra::Centers == 1);

    static constexpr int L = (Bra::L+Ket::L);
    static constexpr int NP = Bra::nherm;
    static constexpr int NQ = nherm2(Ket::L);
    static constexpr int C = Ket::First;
    static constexpr int D = Ket::Second;
    static constexpr int NCD = Ket::nbf;

    using ThreadBlock = cuda::thread_block<DimX,1,DimZ>;
    static constexpr int num_threads = ThreadBlock::size();
    static constexpr int max_shmem = MaxShmem;
    static constexpr int min_blocks = MinBlocks;

    static constexpr int ncd_batch = (NCD+DimZ-1)/DimZ;

    union Registers {
      struct {
        double pCD[NP][ncd_batch];
      };
    };

    struct Shmem {
      Hermite Hx[DimX];
      double R[nherm2(L)][DimX];
      Hermite cd;
      //double Ecd[2][ncd_batch*DimZ];
    };

    __device__
    LIBINTX_GPU_FORCEINLINE
    void operator()(
      const Bra &bra, const int &kab,
      const Ket &ket,
      const auto &boys,
      auto &&args,
      auto &&BraKet) const
    {

      auto &p_orbitals = orbitals(bra);
      auto &q_orbitals = orbitals(ket);

      __shared__ Shmem shmem;

      constexpr ThreadBlock thread_block;
      const int thread_rank = thread_block.thread_rank();
      constexpr auto warp = this_warp();

      int ij = blockIdx.x*DimX;
      int kl = blockIdx.y;

      for (int ix = thread_rank/warp.size(); ix < DimX; ix += num_threads/warp.size()) {
        if (ix+ij < bra.N) {
          memcpy1(bra.hdata(ix+ij,kab), &shmem.Hx[ix], warp);
        }
        else {
          memset1(&shmem.Hx[ix], 0, warp);
          if (warp.thread_rank() == 0) shmem.Hx[ix].exp = 1;
        }
      }

      //fill(NCD*2, (double*)shmem.Ecd[0], 0, thread_block);

      decltype(Registers::pCD) pCD = {};

      for (int kcd = 0; kcd < ket.K; ++kcd) {

        thread_block.sync();

        memcpy(
          nwords<sizeof(double),Hermite>(),
          reinterpret_cast<const double*>(ket.hdata(kl,kcd)),
          reinterpret_cast<double*>(&shmem.cd),
          thread_block
        );

        thread_block.sync();

        //if (!shmem.cd.C) continue;

        const auto &ab = shmem.Hx[threadIdx.x];
        const auto &cd = shmem.cd;

        auto &P = ab.r;
        auto &Q = cd.r;

        auto &R = shmem.R;

        if (threadIdx.z == 0) {
          double p = ab.exp;
          double q = cd.exp;
          double pq = p*q;
          double alpha = pq/(p+q);
          double Ck = ab.C*cd.C;
          Ck *= rsqrt(pq*pq*(p+q));
          double T = alpha*norm(P,Q);
          double s[L+1];
          boys.template compute<L>(T,0,s);
          for (int i = 0; i <= L; ++i) {
            s[i] = math::sqrt_4_pi5*Ck*s[i];
            Ck *= -2*alpha;
          }
          auto PQ = P-Q;
          namespace r1 = libintx::md::r1;
          r1::visit<L,r1::DepthFirst>(
            [&](auto &&r) {
              R[r.index][threadIdx.x] = r.value;
            },
            PQ, s
          );
        }
        thread_block.sync();

#pragma unroll
        for (int iq = 0; iq < NQ-ncart(C+D); ++iq) {
          //auto& Ecd = shmem.Ecd[iq%2];
          auto q = q_orbitals[iq];
          double phase = (q.L()%2 == 0 ?  +1 : -1);
          //memcpy(NCD, ket.gdata(kl,kcd)+iq*NCD, Ecd, thread_block);
          //thread_block.sync();
          //if (threadIdx.x+ij >= bra.N) continue;
#pragma unroll
          for (int icd_batch = 0; icd_batch < ncd_batch; ++icd_batch) {
            int icd = threadIdx.z + icd_batch*DimZ;
            double E = ket.gdata(kl,kcd)[icd + iq*NCD];
#pragma unroll
            for (int ip = 0; ip < NP; ++ip) {
              auto p = p_orbitals[ip];
              double pq = phase*R[herm::index2(p+q)][threadIdx.x];
              pCD[ip][icd_batch] += pq*E;
            }
          }
        }

#pragma unroll
        for (int iq = 0; iq < ncart(C+D); ++iq) {
          //auto& Ecd = shmem.Ecd[iq%2];
          auto q = q_orbitals[nherm2(C+D-1)+iq];
          double inv_2_q = ((C+D)%2 == 0 ?  +1 : -1)*shmem.cd.inv_2_exp;
#pragma unroll
          for (int icd_batch = 0; icd_batch < ncd_batch; ++icd_batch) {
            int icd = threadIdx.z + icd_batch*DimZ;
            double E = inv_2_q*ket.pure_transform[icd + (iq)*NCD];
#pragma unroll
            for (int ip = 0; ip < NP; ++ip) {
              auto p = p_orbitals[ip];
              double pq = R[herm::index2(p+q)][threadIdx.x];
              pCD[ip][icd_batch] += pq*E;
            }
          }
        }

      } // kcd

      constexpr int X = Bra::L;
      double inv_2_p = shmem.Hx[threadIdx.x].inv_2_exp;

      if (ij+threadIdx.x >= bra.N) return;

      for (int icd_batch = 0; icd_batch < ncd_batch; ++icd_batch) {
        int icd = threadIdx.z + icd_batch*DimZ;
        if (icd >= NCD) break;
        double U[ncart(X)];
        double V[npure(X)] = {};
        // if (kab) {
        //   for (int ix = 0; ix < npure(X); ++ix) {
        //     V[ix] = BraKet(threadIdx.x + ij + ix*bra.N, icd + kl*NCD);
        //   }
        // }
        hermite_to_cartesian<X>(
          inv_2_p,
          [&](auto &&p) -> const double& {
            return pCD[herm::index1(p)][icd_batch];
          },
          [&](auto &&p) -> double& { return U[cart::index(p)]; }
        );
        cartesian_to_pure<X>(
          [&](auto &&x, auto v) {
            int ix = index(x);
            BraKet(threadIdx.x + ij + ix*bra.N, icd + kl*NCD) = v + V[ix];
          },
          [&](auto x) {
            return U[cart::index(x)];
          }
        );
      } // icd_batch

    }

  };


  // [q,ij,x,kl] kernel {DimX->q}
  template<int DimX, int MinBlocks, int X, int Ket, typename Boys>
  __global__
  __launch_bounds__(DimX,MinBlocks)
  static void compute_q_x_kernel(
    const Basis1<X> bra,
    const Basis2<Ket> ket,
    const std::pair<int,int> K,
    const Boys boys,
    auto QX)
  {

    static constexpr int L = bra.L+ket.L;
    static constexpr int NP = bra.nherm;
    static constexpr int NQ = ket.nherm;

    constexpr thread_block<DimX> thread_block;
    constexpr int num_threads = thread_block.size();
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
