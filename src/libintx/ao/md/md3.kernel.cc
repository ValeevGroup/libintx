#include "libintx/ao/md/engine.h"
#include "libintx/ao/md/md3.kernel.h"
#include "libintx/ao/md/basis.h"
#include "libintx/ao/md/hermite.h"
#include "libintx/ao/md/r1.h"
#include "libintx/boys/chebyshev.h"
#include "libintx/simd.h"
#include "libintx/blas.h"

#include "libintx/config.h"
#include "libintx/utility.h"

#ifndef LIBINTX_MD_MD3_KERNEL_BRA
#error LIBINTX_MD_MD3_KERNEL_BRA undefined
#endif

#ifndef LIBINTX_MD_MD3_KERNEL_KET
#error LIBINTX_MD_MD3_KERNEL_KET undefined
#endif

namespace libintx::md::kernel {

  constexpr auto Bra = LIBINTX_MD_MD3_KERNEL_BRA;
  constexpr auto Ket = LIBINTX_MD_MD3_KERNEL_KET;

  static const auto &boys = boys::chebyshev<2*LMAX+XMAX+1,Bra+Ket>();

  template<int X, int Ket, typename T>
  LIBINTX_ALWAYS_INLINE
  static void compute_sx(auto bra, auto ket, auto PQ, auto &&Sx) {
    auto p = bra.exp;
    auto q = ket.exp;
    T pq = p*q;
    T alpha = pq/(p+q);
    T x = alpha*norm(PQ);
    T s[Ket+1] = {};
    boys.template compute<Ket>(x, X, s);
    auto Si = bra.C*ket.C/sqrt(pq*pq*(p+q))*math::sqrt_4_pi5;
    Si *= math::pow<X>(-2*alpha*bra.inv_2_exp);
libintx_unroll(12+1)
    for (int i = 0; i <= Ket; ++i) {
      Sx(i, s[i]*Si);
      //printf("s[%i] = %f\n", i, (double)s[i]);
      Si *= -2*alpha;
    }
  }

  template<int X, int Ket, typename T>
  LIBINTX_ALWAYS_INLINE
  static void compute_r1(
    const Hermite<T> &bra,
    const Hermite<double> &ket,
    auto &&R)
  {
    auto PQ = bra.r - ket.r;
    T r0[X+Ket+1] = {};
    compute_sx<X,Ket,T>(
      bra,ket,PQ,
      [&](auto i, auto si) {
        r0[i+X] = si;
      }
    );
    namespace r1 = libintx::md::r1;
    auto f = [&](auto r) {
      if constexpr (r.L >= X) R[r.index] += r.value;
      //R[r.index] += r.value;
    };
    r1::visit<X+Ket>(f, PQ, r0);
  }

  template<int X, int Ket, typename T>
  LIBINTX_ALWAYS_INLINE
  static void compute_x_q(
    auto &&R,
    auto &&V)
  {
    static constexpr int NX = npure(X);
    using hermite::index1;
    using hermite::index2;
    constexpr auto Qs = hermite::orbitals2<Ket>;
#if (LIBINTX_MD_MD3_KERNEL_BRA+LIBINTX_MD_MD3_KERNEL_KET <= 12)
    libintx_unroll(455)
#endif
    for (int iq = 0; auto q : Qs) {
      constexpr auto xs = cartesian::orbitals<X>();
      T Rq[xs.size()];
      libintx_unroll(28)
      for (int i = 0; auto &p : xs) {
        Rq[i++] = R[index2(p+q)];
      }
      T v[NX] = {};
      constexpr pure::Transform<X> pure_transform;
      foreach2<NX,xs.size()>(
        [&](auto ix, auto ic) {
          constexpr auto Cx = pure_transform.data[ic.value][ix.value];
          if constexpr (!Cx) return;
          auto u = Rq[ic.value];
          v[ix.value] += Cx*u;
        }
      );
      libintx_unroll(13)
      for (int ix = 0; ix < NX; ++ix) {
        V(ix,iq) = v[ix];
      }
      ++iq;
    }
  }


  template<int X, int C, int D, Operator Op, typename Params>
  struct KernelImpl : Kernel<Op,Params> {

    using simd_t = typename Kernel<Op,Params>::simd_t;
    static constexpr int Batch = Kernel<Op,Params>::batch(X,C,D);

    static constexpr int NCD = npure(C,D);
    static constexpr int NQ = nherm2(C+D);
    static constexpr int NX = npure(X);

    void compute(
      const Params& params,
      const HermiteBasis<1,simd_t> &bra,
      const HermiteBasis<2,double> &ket,
      simd_t* __restrict__ V
    );

  };


  template<int X, int C, int D, Operator Op, typename Params>
  void KernelImpl<X,C,D,Op,Params>::compute(
    const Params& params,
    const HermiteBasis<1,simd_t> &bra,
    const HermiteBasis<2,double> &ket,
    simd_t* __restrict__ V)
  {
    using std::sqrt;
    using cartesian::orbitals;
    using cartesian::index;
    using hermite::index1;
    using hermite::index2;

    constexpr int Lanes = KernelImpl::Lanes;
    constexpr int Batch = KernelImpl::Batch;
    libintx_assert(bra.N == Batch);
    static_assert(Op == Coulomb);

    if constexpr (NCD <= LIBINTX_SIMD_REGISTERS-1 && NCD < 25) {

      for (int kp = 0; kp < bra.K; ++kp) {
        for (int kq = 0; kq < ket.K; ++kq) {

          auto &q = *ket.hermite(0,kq);
          auto* __restrict__ H = ket.hermite_to_ao(0,kq);

          simd_t Q = q.inv_2_exp;

          for (int ibatch = 0; ibatch < Batch; ++ibatch) {
            simd_t q_x[NX][NQ] = {};
            auto &p = *bra.hermite(ibatch,kp);
            auto U = [&](auto ix, auto iq) ->auto& {
              return q_x[ix][iq];
            };

            simd_t R[nherm2(X+C+D)] = {};
            compute_r1<X,C+D>(p,q,R);
            compute_x_q<X,C+D,simd_t>(R,U);

            for (int ix = 0; ix < NX; ++ix) {
              simd_t v[NCD] = {};
              constexpr int NQ = nherm2(C+D-1);
              //libintx_unroll(20) // NB do not unroll
              for (int iq = 0; iq < NQ; ++iq) {
                auto q = q_x[ix][iq];
                libintx_unroll(LIBINTX_SIMD_REGISTERS)
                for (int icd = 0; icd < NCD; ++icd) {
                  v[icd] += q*H[icd + iq*NCD];
                }
              }
              hermite_to_pure<C,D>(
                [&](auto q) {
                  return q_x[ix][q.value+NQ];
                },
                [&](auto c, auto d, auto &&u) {
                  int icd = index(c) + index(d)*npure(C);
                  v[icd] += Q*u;
                }
              );
              libintx_unroll(LIBINTX_SIMD_REGISTERS)
              for (int icd = 0; icd < NCD; ++icd) {
                V[(ibatch+ix*Batch) + icd*Batch*NX] += v[icd];
              }
            }
          }
        }
      }
    }

    else {

      for (int kq = 0; kq < ket.K; ++kq) {

        auto &q = *ket.hermite(0,kq);
        auto* __restrict__ H = ket.hermite_to_ao(0,kq);

        simd_t R[Batch][nherm2(Bra+Ket)] = {};

        for (int i = 0; i < Batch; ++i) {
          simd_t r0[X+Ket+1] = {};
          auto PQ = bra.hermite(i,0)->r - q.r;
          for (int kp = 0; kp < bra.K; ++kp) {
            auto &p = *bra.hermite(i,kp);
            compute_sx<Bra,Ket,simd_t>(
              p,q,PQ,
              [&](auto i, auto si) {
                r0[i+X] += si;
              }
            );
          }
          namespace r1 = libintx::md::r1;
          auto f = [&](auto r) {
            if constexpr (r.L >= X) R[i][r.index] += r.value;
            //R[r.index] += r.value;
          };
          r1::visit<X+Ket>(f, PQ, r0);
        }


        union {
          simd_t data[NQ][NX][Batch];
          simd_t data2[NQ][NX*Batch];
        } q_x;

        for (int i = 0; i < Batch; ++i) {
          auto U = [&](auto ix, auto iq) ->auto& {
            return q_x.data[iq][ix][i];
          };
          compute_x_q<X,C+D,simd_t>(R[i],U);
        }

        constexpr int NQ = (
          NCD < 49 ?
          nherm2(C+D-1) :
          nherm2(C+D)
        );

        static auto gemm_kernel = (
          (Lanes*Batch*NX*NCD*NQ < 64*64*32)
          ? blas::make_gemm_kernel<double>(
            blas::NoTranspose, blas::Transpose,
            Lanes*Batch*NX, NCD, NQ,
            1.0, nullptr, Lanes*Batch*NX, nullptr, NCD,
            1.0, nullptr, Lanes*Batch*NX
          )
          : nullptr
        );

        if (gemm_kernel) {
          gemm_kernel->compute(
            (const double*)q_x.data,
            (const double*)H,
            (double*)V
          );
        }
        else {
          blas::gemm(
            blas::NoTranspose, blas::Transpose,
            Lanes*Batch*NX, NCD, NQ,
            1.0,
            (const double*)q_x.data, Lanes*Batch*NX,
            (const double*)H, NCD,
            1.0,
            (double*)V, Lanes*Batch*NX
          );
        }

        if constexpr (NQ != nherm2(C+D)) {
          auto Q = q.inv_2_exp;
          for (int ix = 0; ix < Batch*NX; ++ix) {
            hermite_to_pure<C,D>(
              [&](auto q) {
                return q_x.data2[NQ+(q)][ix];
              },
              [&](auto c, auto d, auto u) {
                int icd = index(c) + index(d)*npure(C);
                V[ix + icd*Batch*NX] += Q*u;
              }
            );
          }
        }

      } // ket.K

    } // gemm


  }


  template<int Bra, int Ket, Operator Op, typename Params>
  std::unique_ptr< Kernel<Op,Params> > make_kernel(int C, int D) {
    std::unique_ptr< Kernel<Op,Params> > kernel;
    jump_table(
      std::make_index_sequence<std::min(Ket+1,LMAX+1)>(),
      C,
      [&](auto C) {
        constexpr int D = Ket-C;
        if constexpr (std::max<int>(C,D) <= LMAX) {
          kernel = std::make_unique< KernelImpl<Bra,C,D,Op,Params> >();
        }
      }
    );
    return kernel;
  }

  using CoulombKernel = Kernel<Coulomb,Coulomb::Operator::Parameters>;

  template // <int Bra, int Ket, Operator Op, typename Params, typename T>
  std::unique_ptr<CoulombKernel> make_kernel<Bra,Ket>(
    int C, int D
  );

}
