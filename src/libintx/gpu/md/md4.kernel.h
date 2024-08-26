#include "libintx/gpu/forward.h"
#include "libintx/gpu/md/basis.h"
#include "libintx/gpu/md/md.kernel.h"
#include "libintx/gpu/api/thread_group.h"

#include "libintx/integral/md/r1/recurrence.h"
#include "libintx/math.h"

namespace libintx::gpu::md::kernel {

  namespace cart = libintx::cartesian;
  namespace herm = libintx::hermite;

  // [ij,ab,cd,kl) kernel {ij->DimX,p->DimZ}
  template<
    typename Bra, typename Ket,
    int DimX, int DimY, int DimZ,
    int MaxShmem,
    int MinBlocks = 2
    >
  struct md4_ab_cd_kernel
  // nb: DimZ,DimY swapped
    : kernel::md_x_cd_kernel_base<2,Bra,Ket,DimX,DimZ,DimY,MaxShmem,MinBlocks>
  {
    static_assert(DimY == 1);
  };

  // (ij,ab,cd,kl) kernel {ij->DimX,kl->DimY}
  template<
    typename Bra, typename Ket,
    int DimX, int DimY,
    int MaxShmem,
    int MinBlocks
    >
  struct md4_ab_cd_kernel<Bra,Ket,DimX,DimY,1,MaxShmem,MinBlocks>
    : kernel::md_v0_kernel_base<Bra,Ket,DimX,DimY,MaxShmem,MinBlocks>
  {
  };


  /// [ij,r1,kl] kernel {ij->DimX,kl->DimY}
  template<int DimX, int DimY, int MinBlocks>
  __global__
  __launch_bounds__(DimX*DimY,MinBlocks)
  static void compute_r1_kernel(
    auto bra, auto ket,
    std::pair<int,int> K,
    auto boys,
    auto R1)
  {

    static constexpr int L = (bra.L+ket.L);
    static constexpr int NH = nherm2(L);

    constexpr gpu::thread_block<DimX,DimY> thread_block;
    constexpr auto warp = this_warp();
    constexpr int num_threads = thread_block.size();
    int thread_rank = thread_block.thread_rank();

    static_assert(num_threads%warp.size() == 0);

    __shared__ Hermite abs[thread_block.x];
    __shared__ Hermite cds[thread_block.y];

    for (int i = thread_rank/warp.size(); i < thread_block.x; i += num_threads/warp.size()) {
      int idx = i + blockIdx.x*thread_block.x;
      if (idx >= bra.N) {
        memset1(&abs[i], 0, warp);
        continue;
      }
      memcpy1(bra.hdata(idx,K.first), &abs[i], warp);
    }

    for (int i = thread_rank/warp.size(); i < thread_block.y; i += num_threads/warp.size()) {
      int idx = i + blockIdx.y*thread_block.y;
      if (idx >= ket.N) {
        memset1(&cds[i], 0, warp);
        continue;
      }
      memcpy1(ket.hdata(idx,K.second), &cds[i], warp);
    }

    thread_block.sync();

    const auto &ab = abs[threadIdx.x];
    const auto &cd = cds[threadIdx.y];

    auto &P = ab.r;
    auto &Q = cd.r;
    auto PQ = P-Q;

    double C = ab.C*cd.C;
    if (!C) return;

      //C *= 2*std::pow(M_PI,2.5);
    double s[L+1] = {};

    double p = ab.exp;
    double q = cd.exp;
    double pq = p*q;
    double alpha = pq/(p+q);
    double T = alpha*norm(P,Q);
    boys.template compute<L>(T, 0, s);
    C *= rsqrt(pq*pq*(p+q));
    //double Kab = exp(-(a*b)/p*norm(P));
    //double Kcd = 1;//exp(-norm(Q));
    //C *= Kcd;
    for (int i = 0; i <= L; ++i) {
      s[i] *= math::sqrt_4_pi5*C;
      //printf("s[%i]=%f\n", i, s[i]);
      C *= -2*alpha;
    }

    auto v = [&](auto r) {
      int kl = threadIdx.y + DimY*blockIdx.y;
      R1(threadIdx.x, r.index, blockIdx.x, kl) = r.value;
      //printf("r1[%i]=%f\n", threadIdx.x + r.index*thread_block.x + ridx, r.value);
    };

    namespace r1 = libintx::md::r1;
    r1::visit<L,r1::DepthFirst>(v, PQ, s);

  }


  // [p,ij,q,kl] kernel {p->DimX,q->DimY}
  template<int DimX, int DimY, int MinBlocks, int Bra, int Ket, typename Boys>
  __global__
  __launch_bounds__(DimX*DimY,MinBlocks)
  static void compute_p_q_kernel(
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

    constexpr thread_block<DimX,DimY> thread_block;
    constexpr int num_threads = thread_block.size();
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


  // [ij,p,cd,kl) kernel {ij->DimX,p->DimY}
  template<
    typename Bra, typename Ket,
    int DimX, int DimY,
    int MaxShmem,
    int MinBlocks = 2
    >
  struct md4_v1_p_cd_kernel
    : kernel::md_x_cd_kernel_base<0,Bra,Ket,DimX,DimY,1,MaxShmem,MinBlocks>
  {
  };


  // [ij,r1,kl] -> [ij,p,cd,kl) kernel {ij->DimX,p->DimY}
  template<
    typename Bra, typename Ket,
    int DimX, int DimY,
    int MaxShmem,
    int MinBlocks = 2
    >
  struct md4_v1_r1_p_cd_kernel {

    using ThreadBlock = gpu::thread_block<DimX,DimY>;
    static constexpr int num_threads = ThreadBlock::size();
    static constexpr int max_shmem = MaxShmem;
    static constexpr int min_blocks = MinBlocks;

    static constexpr int L = (Bra::L + Ket::L);
    static constexpr int NH = nherm2(L);
    static constexpr int NQ = nherm2(Ket::L-1);
    static constexpr int NC = npure(Ket::First);
    static constexpr int ND = npure(Ket::Second);
    static constexpr int NCD = NC*ND;

    // min shared memory, actual allocation is dynamic
    struct Shmem {
      double inv_2_q;
      double Ecd[NCD*NQ];
    };

    struct Registers {
      double V[NCD];
    };

    __device__
    static Shmem* shmem() {
      extern __shared__ double shmem[];
      return reinterpret_cast<Shmem*>(shmem);
    }

    __device__ LIBINTX_GPU_FORCEINLINE
    void operator()(
      const Bra &bra,
      const Ket &ket,
      int kcd, int nk,
      auto &R1, auto &pCD)
    {

      using hermite::index2;

      const auto &p_orbitals = orbitals(bra);
      const auto &q_orbitals = orbitals(ket);

      constexpr int NP = bra.nherm;
      constexpr ThreadBlock thread_block;
      //static_assert(thread_block.y == 1);

      int kl = blockIdx.y;
      assert(kl < ket.N);

      Shmem *shmem = this->shmem();

      for (int k = 0; k < nk; ++k) {
        memcpy(1+NQ*NCD, ket.gdata(kl,k+kcd)-1, (double*)&shmem[k], thread_block);
        if (thread_block.thread_rank() == 0) {
          constexpr int phase = (Ket::L%2 == 0 ? +1 : -1);
          shmem[k].inv_2_q *= phase;
        }
      }
      thread_block.sync();

      for (int ip = threadIdx.y; ip < NP; ip += thread_block.y) {

        auto p = p_orbitals[ip];
        decltype (Registers::V) V = {};

        if (kcd != 0) {
#pragma unroll
          for (int icd = 0; icd < NCD; ++icd) {
            V[icd] = pCD(threadIdx.x, ip, icd, kl, blockIdx.x);
          }
        }

        for (int k = 0; k < nk; ++k) {

          auto &Ecd = shmem[k].Ecd;
          auto inv_2_q = shmem[k].inv_2_q;

          //#pragma unroll
          for (int iq = 0; iq < NQ; ++iq) {
            auto q = q_orbitals[iq];
            auto pq = R1(threadIdx.x, index2(p+q), blockIdx.x, kl, k);
            pq *= (q.L()%2 == 0 ? +1 : -1);
#pragma unroll
            for (int i = 0; i < NCD; ++i) {
              V[i] += Ecd[i + iq*NCD]*pq;
            }
          }

          double r[ncart(Ket::L)];
          for (int i = 0; i < ncart(Ket::L); ++i) {
            auto q = q_orbitals[i+NQ];
            r[i] = R1(threadIdx.x, index2(p+q), blockIdx.x, kl, k);
          }

          hermite_to_pure<Ket::First,Ket::Second>(
            [&](auto c, auto d, auto u) {
              V[index(c) + index(d)*NC] += inv_2_q*u;
            },
            [&](auto q) {
              assert(index2(p+q) < NH);
              return r[cartesian::index(q)];
              //return R1(threadIdx.x, index2(p+q), blockIdx.x, kl, k);
            }
          );

        }

        for (int icd = 0; icd < NCD; ++icd) {
          //printf("
          pCD(threadIdx.x, ip, icd, kl, blockIdx.x) = V[icd];
        }

      }

    }

  };


  // [ij,p,cd,kl) -> [ij,ab,cd,kl) kernel {ij->DimX,cd*kl->DimY}
  template<
    typename Bra, typename Ket,
    int DimX, int DimY,
    int MaxShmem,
    int MinBlocks = 2
    >
  struct md4_v1_ab_cd_kernel {

    using ThreadBlock = gpu::thread_block<DimX,DimY>;
    static constexpr int num_threads = ThreadBlock::size();
    static constexpr int max_shmem = MaxShmem;
    static constexpr int min_blocks = MinBlocks;

    static constexpr int NP = nherm2(Bra::L-1);
    static constexpr int NA = npure(Bra::First);
    static constexpr int NB = npure(Bra::Second);
    static constexpr int NAB = NA*NB;

    static constexpr int np_batch = []() {
      if (NP <= 1) return 1;
      for (auto n : { 4, 3, 2 }) {
        if (NP%n != 0) continue;
        if (n*NAB*DimX*sizeof(double) <= MaxShmem) return n;
      }
      return 1;
    }();

    union Shmem {
      double Eab[np_batch*NAB][DimX];
    };

    struct Registers {
      double V[NAB];
      //double p[ncart(A+B)]; exclude from reg count
    };

    __device__ LIBINTX_GPU_FORCEINLINE
    void operator()(
      size_t Nij, size_t Nkl,
      auto &&ABp,
      auto &&pX,
      double Ck,
      auto &&ABX,
      auto *hermite_to_pure_transform)
    {

      constexpr auto thread_block = ThreadBlock();

      int ij = threadIdx.x + blockIdx.x*thread_block.x;
      int kl = threadIdx.y + blockIdx.y*thread_block.y;

      decltype (Registers::V) V = {};

      if (Ck && (ij < Nij && kl < Nkl)) {
        for (int iab = 0; iab < NAB; ++iab) {
          //printf("V[%i,%i] = %f\n", ij, kl, V[iab]);
          V[iab] = ABX(ij+iab*Nij,kl);
        }
      }

      if (ij < Nij && kl < Nkl) {
        double C = ABp(threadIdx.x,0,blockIdx.x);
        if constexpr (!hermite_to_pure_too_complicated(Bra::First,Bra::Second)) {
          //decltype (Registers::p) p = {};
          double r[ncart(Bra::L)] = {}; // exclude from regs for now
          for (int ip = 0; ip < ncart(Bra::L); ip += 1) {
            r[ip] = pX(threadIdx.x,ip+NP,kl,blockIdx.x);
            //printf("pX(%i,%i,%i,%i)=%f\n", threadIdx.x, ip+NP, kl, blockIdx.x, p[ip]);
          }
          hermite_to_pure<Bra::First,Bra::Second>(
            [&](auto &&i, auto &&j, auto &&u) {
              int iab = index(i) + index(j)*NA;
              V[iab] += C*u;
            },
            [&](auto &&p) {
              return r[cart::index(p)];
            }
          );
        }
        else {
          for (int ip = 0; ip < ncart(Bra::L); ++ip) {
            auto p = C*pX(threadIdx.x,ip+NP,kl,blockIdx.x);
            for (int iab = 0; iab < NAB; ++iab) {
              V[iab] += p*hermite_to_pure_transform[iab + ip*NAB];
            }
          }
        } // !hermite_to_pure_too_complicated

      }

      __shared__ Shmem shmem;
      auto &Eab = shmem.Eab;
      for (int ip = 0; ip < NP; ip += np_batch) {
        thread_block.sync();
        memcpy(np_batch*DimX*NAB, &ABp(0,1+ip*NAB,blockIdx.x), &Eab[0][0], thread_block);
        thread_block.sync();
        double p[np_batch];
        for (int i = 0; i < np_batch; ++i) {
          p[i] = pX(threadIdx.x,ip+i, kl, blockIdx.x);
          //printf("pX(%i,%i,%i,%i)=%f\n", threadIdx.x, ip+i, kl, blockIdx.x, p[i]);
        };
#pragma unroll
        for (int iab = 0; iab < NAB; ++iab) {
          for (int k = 0; k < np_batch; ++k) {
            double h = Eab[iab+k*NAB][threadIdx.x];
            V[iab] += h*p[k];
          }
        }
      }

      if (ij >= Nij || kl >= Nkl) return;

      for (int iab = 0; iab < NAB; ++iab) {
        //printf("V[%i,%i] = %f\n", ij, kl, V[iab]);
        ABX(ij+iab*Nij,kl) = V[iab];
      }

    }

  };

  // [p,cd,kl,ij] kernel {p->DimX}
  template<
    typename Bra, typename Ket,
    int DimX,
    int MaxShmem,
    int MinBlocks = 2
    >
  struct md4_v2_p_cd_kernel {

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
      } static_;
      struct Dynamic {
        Hermite cd;
        double Ecd[NQ][NCD];
        double R[nherm2(L)];
      } dynamic_;
    };

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

          if constexpr (!hermite_to_pure_too_complicated(Ket::First,Ket::Second)) {
            constexpr int phase = (Ket::L%2 == 0 ? +1 : -1);
            for (int k = 0; k < nk; ++k) {
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
          }
          else {
            constexpr int phase = (Ket::L%2 == 0 ? +1 : -1);
            for (int iq = 0; iq < ncart(Ket::L); ++iq) {
              const auto q = orbitals2[NQ+iq];
              int ipq = herm::index2(p+q);
              for (int k = 0; k < nk; ++k) {
                double inv_2_q = phase*shmem[k].cd.inv_2_exp;
                double r = inv_2_q*shmem[k].R[ipq];
                for (int icd = 0; icd < NCD; ++icd) {
                  auto C = bra.pure_transform[icd + iq*NCD];
                  V[icd] += r*C;
                }
              }
            }
          } // hermite_to_pure_too_complicated

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
