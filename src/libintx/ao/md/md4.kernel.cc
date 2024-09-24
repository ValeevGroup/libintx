// -*-c++-*-

#include "libintx/ao/md/engine.h"
#include "libintx/ao/md/basis.h"
#include "libintx/ao/md/hermite.h"
#include "libintx/ao/md/md4.kernel.h"
#include "libintx/boys/chebyshev.h"
#include "libintx/blas.h"
#include "libintx/simd.h"

#include "libintx/utility.h"
#include "libintx/config.h"

#if (LIBINTX_MD_MD4_KERNEL_KET <= 3)
#define LIBINTX_AO_MD_R1_KERNEL_INLINE LIBINTX_ALWAYS_INLINE
#endif
#include "libintx/ao/md/r1/kernel.h"

#ifndef LIBINTX_MD_MD4_KERNEL_BRA
#error LIBINTX_MD_MD4_KERNEL_BRA undefined
#endif

#ifndef LIBINTX_MD_MD4_KERNEL_KET
#error LIBINTX_MD_MD4_KERNEL_KET undefined
#endif

namespace libintx::md::kernel {

  constexpr struct {
    int bra = 8;
  } Optimize;

  constexpr auto Bra = LIBINTX_MD_MD4_KERNEL_BRA;
  constexpr auto Ket = LIBINTX_MD_MD4_KERNEL_KET;

  constexpr auto p_orbitals = hermite::orbitals2<Bra>;
  constexpr auto q_orbitals = hermite::orbitals2<Ket>;

  template<int NP, int NQ, int IQ = 0>
  //LIBINTX_ALWAYS_INLINE
  void compute_p_q(auto &&R, auto &&pq) {
    if constexpr (NP*NQ <= 84*84) {
libintx_unroll (455)
      for (int iq = 0; iq < NQ; ++iq) {
        auto q = q_orbitals[IQ+iq];
libintx_unroll (455)
        for (int ip = 0; ip < NP; ++ip) {
          auto p = p_orbitals[ip];
          pq[iq][ip] = R[hermite::index2(p+q)];
        }
      }
    }
    else {
      for (int iq = 0; iq < NQ; ++iq) {
        auto q = q_orbitals[IQ+iq];
        //libintx_unroll (455)
        for (int ip = 0; ip < NP; ++ip) {
          auto p = p_orbitals[ip];
          pq[iq][ip] = R[hermite::index2(p+q)];
        }
      }
    }
  }

  template<typename T, int C, int D>
  LIBINTX_ALWAYS_INLINE
  void hermite_to_pure(auto &&R, auto &&pcd) {
    constexpr int NP = nherm2(Bra);
    constexpr int NQ = ncart(Ket);
    constexpr int NCD = npure(C,D);
    static_assert(p_orbitals.size() >= NP);

    if constexpr (NCD <= 49 && Bra <= Optimize.bra) {
      constexpr auto pure_transform = md::pure_transform<C,D>();
      constexpr int Tile = LIBINTX_SIMD_REGISTERS-1;
      //for (int ip_batch = 0; ip_batch < (NP+Tile-1)/Tile; ++ip_batch) {
      foreach<(NP+Tile-1)/Tile>(
        [&](auto ip_batch) {
        constexpr int ip_tile = ip_batch*Tile;
        T pq[NQ][Tile] = {};
        libintx_unroll (455)
        for (size_t iq = 0; iq < NQ; ++iq) {
          auto q = q_orbitals[iq+nherm2(Ket-1)];
          libintx_unroll (LIBINTX_SIMD_REGISTERS)
          for (int ip = 0; ip < Tile; ++ip) {
            if (ip+ip_tile >= NP) break;
            auto p = p_orbitals[ip+ip_tile];
            auto r = R[hermite::index2(p+q)];
            pq[iq][ip] = r;
          }
        }
        foreach(
          std::make_index_sequence<NCD>{},
          [&](auto icd) {
            constexpr int ic = icd%npure(C);
            constexpr int id = icd/npure(C);
            T p_icd[Tile] = {};
            foreach(
              std::make_index_sequence<NQ>{},
              [&](auto iq) {
                constexpr auto c_lm = pure_transform.data[iq.value][id][ic];
                if constexpr (!c_lm) return;
                //libintx_unroll(LIBINTX_SIMD_REGISTERS)
                for (size_t ip = 0; ip < Tile; ++ip) {
                  p_icd[ip] += c_lm*pq[iq][ip];
                }
              }
            );
            for (size_t ip = 0; ip < Tile; ++ip) {
              if (ip+ip_tile >= NP) break;
              pcd[icd][ip+ip_tile] += p_icd[ip];
            }
          }
        );
        }
      );
    }
    else {
      static auto pure_transform = md::pure_transform<C,D>();
      T pq[NQ][NP];
      compute_p_q<NP,NQ,nherm2(Ket-1)>(R,pq);
      size_t M = NP;
      if constexpr (is_simd_v<T>) M *= simd::size<T>;
      blas::gemm(
        blas::NoTranspose, blas::Transpose,
        M, NCD, NQ,
        1.0,
        reinterpret_cast<const double*>(pq), M,
        reinterpret_cast<const double*>(pure_transform.data), NCD,
        1.0, reinterpret_cast<double*>(pcd), M
      );
    }
  }

  template<typename T, int M, int N, int K>
  auto& transform1_gemm_kernel() {
    constexpr int Lanes = simd::size<T>;
    constexpr int Flops = Lanes*M*N*K;
    static std::shared_ptr< blas::GemmKernel<double> > gemm_kernel = (
      (Flops < Lanes*64*64*32)
      ? blas::make_gemm_kernel<double>(
          blas::NoTranspose, blas::Transpose,
          Lanes*M, N, K,
          1.0, nullptr, Lanes*M, nullptr, N,
          1.0, nullptr, Lanes*M
      )
      : nullptr
    );
    return gemm_kernel;
  }

  template<int C, int D, typename T>
  LIBINTX_ALWAYS_INLINE
  void transform1(
    const auto* __restrict__ R1,
    const auto* __restrict__ Hcd,
    auto &&pq,
    auto &&pCD,
    blas::GemmKernel<double> *gemm_kernel)
  {

    static constexpr int Ket = (C+D);
    static constexpr int NP = nherm2(Bra);
    static constexpr int NQ = nherm2(Ket-1);
    static constexpr int NCD = npure(C,D);

    using hermite::index2;

    if constexpr (!NQ) return;

    // if Hcd fits into regs with at least 5 regs for p
    if constexpr (Bra <= Optimize.bra && LIBINTX_SIMD_REGISTERS-NQ-1 >= 5) {

      constexpr int Tile = std::min(NP,LIBINTX_SIMD_REGISTERS-NQ-1);

      for (int icd = 0; icd < NCD; ++icd) {

        T h[NQ];
        for (int iq = 0; iq < NQ; ++iq) {
          h[iq] = Hcd[icd + iq*NCD];
        }

        libintx_unroll (455)
        for (int ip_batch = 0; ip_batch < (NP+Tile-1)/Tile; ++ip_batch) {

          int ip_tile = ip_batch*Tile;
          T U[Tile] = {};

          libintx_unroll (LIBINTX_SIMD_REGISTERS) // Max
          for (int iq = 0; iq < NQ; ++iq) {
            libintx_unroll (LIBINTX_SIMD_REGISTERS) // Max
            for (int ip = 0; ip < Tile; ++ip) {
              if (ip+ip_tile >= NP) break;
              auto q = q_orbitals[iq];
              auto p = p_orbitals[ip+ip_tile];
              auto pq = R1[index2(p+q)];
              U[ip] += h[iq]*pq;
            }
          }

          libintx_unroll (LIBINTX_SIMD_REGISTERS)
          for (int ip = 0; ip < Tile; ++ip) {
            if (ip+ip_tile >= NP) break;
            pCD[icd][ip+ip_tile] += U[ip];
          }

        }


      } // ip_tile

    } // (NCD < 25)

    else {

      compute_p_q<NP,NQ>(R1, pq);

      if (gemm_kernel) {
        gemm_kernel->compute(
          reinterpret_cast<const double*>(pq),
          Hcd,
          reinterpret_cast<double*>(pCD)
        );
      }

      else {
        // [p|cd] = [p|q]*H(cd,q)'
        size_t M = NP;
        if (is_simd_v<T>) M *= simd::size<T>;
        blas::gemm(
          blas::NoTranspose, blas::Transpose,
          M, NCD, NQ,
          1.0,
          reinterpret_cast<double*>(pq), M,
          Hcd, NCD,
          1.0, reinterpret_cast<double*>(pCD), M
        );
      }

    }

  }

  template<int A, int B, typename T>
  //LIBINTX_ALWAYS_INLINE
  void transform2(int ncd, auto &&bra, auto &&E, const auto &pCD, T* __restrict__ ABCD) {

    constexpr bool use_hermite_to_pure = (Bra <= Optimize.bra);
    static constexpr int NAB = npure(A,B);
    static constexpr int NP = nherm2(A+B-use_hermite_to_pure);

    if constexpr (std::is_scalar_v<T> && (NAB >= 25 || (Bra > Optimize.bra))) {
      blas::gemm(
        blas::NoTranspose, blas::NoTranspose,
        NAB, ncd, NP,
        1.0,
        reinterpret_cast<const double*>(E), NAB,
        reinterpret_cast<const double*>(pCD), NP,
        1.0, reinterpret_cast<double*>(ABCD), NAB
      );
      goto hermite_to_pure;
    }

    static constexpr struct {
      int ab = (
        NAB < LIBINTX_SIMD_REGISTERS
        ? NAB
        : (LIBINTX_SIMD_REGISTERS-3-1)/3
      );
      int cd = std::max(1, (LIBINTX_SIMD_REGISTERS-1)/(ab+1));
      int p = NP/std::max(1, ((NP+60-1)/60));
    } tile;

    static_assert(tile.cd == 1 || tile.ab*tile.cd + 1 + tile.cd <= LIBINTX_SIMD_REGISTERS);
    static_assert(tile.p || !NP);

    for (int ip_tile = 0; ip_tile < NP; ip_tile += tile.p) {

      libintx_unroll (13*13)
      for (int iab_tile = 0; iab_tile < NAB; iab_tile += tile.ab) {

        // NB: NCD < NAB generally, favour E(ab,p) with outer ab loop
        for (int icd_tile = 0; icd_tile < ncd; icd_tile += tile.cd) {

          T V[tile.cd][tile.ab] = {};

          //libintx_unroll (455) // NB dont unroll explicitly
          for (int ip = 0; ip < tile.p; ++ip) {
            if (ip+ip_tile >= NP) break;
            T p[tile.cd] = {};
            for (int icd = 0; icd < tile.cd; ++icd) {
              if (icd+icd_tile >= ncd) break;
              p[icd] = pCD[icd+icd_tile][ip+ip_tile];
            }
            libintx_unroll (LIBINTX_SIMD_REGISTERS)
            for (int iab = 0; iab < tile.ab; ++iab) {
              if (iab+iab_tile >= NAB) break;
              auto e = E[(iab+iab_tile) + (ip+ip_tile)*NAB];
              libintx_unroll (LIBINTX_SIMD_REGISTERS)
              for (int icd = 0; icd < tile.cd; ++icd) {
                V[icd][iab] += e*p[icd];
              }
            }
          }

          for (int icd = 0; icd < tile.cd; ++icd) {
            if (icd+icd_tile >= ncd) break;
            for (int iab = 0; iab < tile.ab; ++iab) {
              if (iab_tile + iab >= NAB) break;
              ABCD[(iab+iab_tile) + (icd+icd_tile)*NAB] += V[icd][iab];
            } // iab
          } // icd

        } // icd_tile
      } // iab_tile

    }

    [[maybe_unused]]
  hermite_to_pure:

    // hermite to pure sparse term
    if constexpr (use_hermite_to_pure) {
      constexpr auto pure_transform = md::pure_transform<A,B>();
      for (int icd = 0; icd < ncd; ++icd) {
        libintx_unroll (13*13)
        for (int iab = 0; iab < NAB; ++iab) {
          T v = {};
          constexpr int N = ncart(A+B);
          libintx_unroll (91)
          for (size_t ip = 0; ip < N; ++ip) {
            auto p = pCD[icd][ip+nherm2(A+B-1)];
            int ia = (iab)%npure(A);
            int ib = (iab)/npure(A);
            auto c_lm = pure_transform.data[ip][ib][ia];
            if (!c_lm) continue;
            v += c_lm*p;
          }
          ABCD[iab + icd*NAB] += bra.inv_2_exp*v;
        }
      }
    }

  }


  template<Operator Op, typename Parameters, int A, int B, int C, int D>
  struct Kernel<Op,Parameters,A,B,C,D> : Kernel<Op,Parameters> {
    using simd_t = typename Kernel<Op,Parameters>::simd_t;

    static_assert(kernel::Bra == A+B);
    static constexpr auto Batch = Kernel::batch(A,B,C,D);

    static constexpr int NP = nherm2(A+B);
    static constexpr int NAB = npure(A,B);
    static constexpr int NCD = npure(C,D);

    Kernel() {
      // may or may not be available or needed
      gemm_kernel_ = transform1_gemm_kernel<simd_t,NP,NCD,nherm2(C+D-1)>().get();
    }

    //LIBINTX_ALWAYS_INLINE
    void compute_p_cd(
      const Parameters&,
      const HermiteBasis<2,simd_t> &bra,
      const HermiteBasis<2,double> &ket,
      const std::function<void(int,int,simd_t(&)[],int)> &V,
      double precision) override
      //auto &&V)
    {

      const auto &boys = boys::chebyshev<4*LMAX+1>();

      // libintx_assert(bra.N == batch.bra);
      libintx_assert(ket.N <= Batch.ket);

      for (int ij = 0; ij < (bra.N+bra.Lanes-1)/bra.Lanes; ++ij) {
        //printf("bra.N=%i, ij=%i\n", bra.N, ij);
        for (int kab = 0; kab < bra.K; ++kab) {

          const auto &p = *bra.hermite(ij,kab);

          // [p,cd,kl)
          for (int kl = 0; kl < ket.N; ++kl) {
            simd_t Rk[nherm2(A+B+C+D)] = {};
            std::fill(
              (simd_t*)std::begin(this->batch_ncd_np_3[kl]),
              (simd_t*)std::end(this->batch_ncd_np_3[kl]),
              simd_t()
            );
            for (int kcd = 0; kcd < ket.K; ++kcd) {
              const auto &q = *ket.hermite(kl,kcd);
              //if (p.norm*q.norm < precision) continue;
              //continue;
              auto* Ecd = ket.hermite_to_ao(kl,kcd);
              simd_t R1[nherm2(A+B+C+D)] = {};
              //auto &R1 = this->R1;
              auto Cpq = p.C*q.C;
              md::r1::compute<A+B+C+D,simd_t>(p.exp, q.exp, p.r-q.r, Cpq, boys, R1);
              auto inv_2_q = q.inv_2_exp;
              for (size_t iq = nherm2(C+D-1); iq < nherm2(A+B+C+D); ++iq) {
                Rk[iq] += inv_2_q*R1[iq];
              }
              md::kernel::transform1<C,D,simd_t>(
                R1, Ecd, this->pq, this->batch_ncd_np_3[kl],
                this->gemm_kernel_
              );
            }
            hermite_to_pure<simd_t,C,D>(Rk, this->batch_ncd_np_3[kl]);
          }

          //const auto &p_cd_batch = this->batch_ncd_np_2;
          V(ij, kab, this->batch_ncd_np_1, NCD*ket.N);

        } // kab
      } // ij

    }

    void compute(
      const Parameters &params,
      const HermiteBasis<2,simd_t> &bra,
      const HermiteBasis<2,double> &ket,
      simd_t* __restrict__ V) override
    {

      //printf("Bra.N=%i, Batch.bra=%i\n", bra.N, Batch.bra);
      libintx_assert(bra.N <= Batch.bra*Kernel::Lanes);
      libintx_assert(ket.N <= Batch.ket);

      // (ab,cd,kl) += E(ab,p)*[p,cd,kl)
      auto V1 = [&](int ij, int kab, simd_t (&p_cd_batch)[], int ncd) {
        //printf("idx=%i,%i, dims=%i,%i\n", ij, kl, nij, nkl);
        const auto *E = bra.hermite_to_ao(ij,kab);
        const auto &p = *bra.hermite(ij,kab);
        transform2<A,B,simd_t>(
          ncd, p, E, reinterpret_cast<const simd_t(&)[][NP]>(batch_ncd_np_2),
          V+ij
        );
      };

      this->compute_p_cd(params, bra, ket, V1, 1e-10);

    }

  private:

    blas::GemmKernel<double>* gemm_kernel_;

    simd_t R1[nherm2(A+B+C+D)] = {};
    simd_t Rq[nherm2(A+B+C+D)] = {};
    simd_t pq[nherm2(C+D-1)][NP] = {};
    union {
      simd_t batch_ncd_np_1[(Batch.ket*(NCD)+2)*NP]; // NB padded for transform2 tile
      simd_t batch_ncd_np_2[Batch.ket*(NCD)][NP];
      simd_t batch_ncd_np_3[Batch.ket][NCD][NP];
    };

  };

  template<int Bra, int Ket, Operator Op, typename Parameters>
  std::unique_ptr< Kernel<Op,Parameters> > make_kernel(int a, int b, int c, int d) {
    static_assert(Bra == kernel::Bra);
    std::unique_ptr< Kernel<Op,Parameters> > kernel;
    jump_table(
      std::make_index_sequence<std::min(Bra,LMAX)+1>{},
      std::make_index_sequence<std::min(Ket,LMAX)+1>{},
      a, c,
      [&](auto A, auto C) {
        constexpr int B = Bra-A;
        constexpr int D = Ket-C;
        if constexpr (std::max<int>(A,B) <= LMAX && std::max<int>(C,D) <= LMAX) {
          kernel = std::make_unique< Kernel<Op,Parameters,A,B,C,D> >();
        }
      }
    );
    libintx_assert(kernel);
    return kernel;
  }


  using CoulombKernel = Kernel<Coulomb,Coulomb::Operator::Parameters>;

  template
  std::unique_ptr<CoulombKernel> make_kernel<Bra,Ket>(
    int,int,int,int
  );

}
