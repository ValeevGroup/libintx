#include "libintx/cuda/forward.h"
#include "libintx/cuda/md/basis.h"
#include "libintx/cuda/md/md.kernel.h"
#include "libintx/cuda/api/thread_group.h"

#include "libintx/engine/md/r1/recurrence.h"
#include "libintx/math.h"

namespace libintx::cuda::md::kernel {

  namespace cart = libintx::cartesian;
  namespace herm = libintx::hermite;

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
      R[rank] = math::sqrt_4_pi5*C*Fm;
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
      const auto q = orbitals2[iq];
      int phase = (q.L()%2 == 0 ? +1 : -1);
      for (int ip = threadIdx.x; ip < NP; ip += thread_block.x) {
        const auto p = orbitals2[ip];
        auto ipq = herm::index2(p+q);
        H(ip,blockIdx.x,iq,blockIdx.y) = phase*R[ipq];
      }
    }

  }


  template<
    typename Bra, typename Ket,
    int DimX,
    int MaxShmem,
    int MinBlocks = 2
    >
  struct md_v2_p_cd_kernel {

    static constexpr int num_threads = DimX;
    static constexpr int min_blocks = MinBlocks;

    static constexpr int L = Bra::L+Ket::L;
    static constexpr int NP = Bra::nherm;
    static constexpr int NQ = nherm2(Ket::L-1);
    static constexpr int NCD = Ket::nbf;

    using ThreadBlock = thread_block<DimX,1,1>;

    struct Registers {
      double V[NCD];
    };

    struct Shmem {
      struct Static {
        Hermite ab;
        array<double,3> PQ;
      };
      struct Dynamic {
        Hermite cd;
        double Ecd[NQ][NCD];
        double R[nherm2(L)];
      };
    };

    // compute [p,q,kl,ij]
    __device__
    void operator()(
      const Bra &bra, int kab,
      const Ket &ket, int k_batch,
      const auto &boys,
      auto &pCD)
    {

      using cartesian::orbital;
      using cartesian::index;

      constexpr ThreadBlock thread_block;
      constexpr int num_threads = ThreadBlock::size();
      int rank = thread_block.thread_rank();

      __shared__ Hermite ab;
      __shared__ array<double,3> PQ;

      extern __shared__ double dynamic_shmem[];
      auto *shmem = reinterpret_cast<typename Shmem::Dynamic*>(dynamic_shmem);

      memcpy1(bra.hdata(blockIdx.x,kab), &ab, thread_block);

      for (int kcd = 0; kcd < ket.K; kcd += k_batch) {

        thread_block.sync();

        int nk = min(k_batch,ket.K-kcd);

        for (int k = 0; k < nk; ++k) {

          memcpy(
            (sizeof(shmem->cd) + sizeof(shmem->Ecd))/sizeof(double),
            (double*)ket.hdata(blockIdx.y,kcd+k),
            (double*)&shmem[k],
            thread_block
          );
          thread_block.sync();

          auto &cd = shmem[k].cd;
          auto &Q = cd.r;
          auto &P = ab.r;

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
            shmem[k].R[rank] = math::sqrt_4_pi5*C*Fm;
            //printf("ip=%i vc=%f\n", rank, R[rank]);
            //printf("T=%f (0)=%f\n", T, R[rank]);
          }

          if (rank < 3) PQ[rank] = P[rank] - Q[rank];
          thread_block.sync();

          if constexpr (L > 0) {
            namespace r1 = libintx::md::r1;
            r1::compute<L>(r1::recurrence, PQ, shmem[k].R, thread_block);
          }

        } // k

        thread_block.sync();

        for (int ip = 0; ip < NP; ip += thread_block.x) {
          if (ip+threadIdx.x >= NP) break;

          const auto p = orbitals2[ip+threadIdx.x];
          double V[NCD] = {};

          if (kcd) {
            for (int icd = 0; icd < NCD; ++icd) {
              V[icd] = pCD(ip+threadIdx.x,icd,blockIdx.y,blockIdx.x);
            }
          }

          for (int k = 0; k < nk; ++k) {
            constexpr int phase = (Ket::L%2 == 0 ? +1 : -1);
            double inv_2_q = phase*shmem[k].cd.inv_2_exp;
            hermite_to_pure<Ket::First,Ket::Second>(
              [&](auto &&c, auto &&d, auto &&v) {
                int icd = index(c) + index(d)*npure(Ket::First);
                V[icd] += inv_2_q*v;
              },
              [&](auto &&q) {
                return shmem[k].R[herm::index2(p+q)];
              }
            );
          }

          for (int iq = 0; iq < NQ; ++iq) {
            const auto q = orbitals2[iq];
            int phase = (q.L()%2 == 0 ? +1 : -1);
            auto ipq = herm::index2(p+q);
            for (int k = 0; k < nk; ++k) {
              double r = phase*shmem[k].R[ipq];
              //H(ip,iq,blockIdx.y,blockIdx.x) =
              for (int icd = 0; icd < NCD; ++icd) {
                V[icd] += r*shmem[k].Ecd[iq][icd];
              }
            }
          }

          for (int icd = 0; icd < NCD; ++icd) {
            pCD(ip+threadIdx.x,icd,blockIdx.y,blockIdx.x) = V[icd];
          }

        } // ip

      }

    }

  };

}
