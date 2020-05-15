#ifndef LIBINTX_ENGINE_OS_RECURRENCE_H
#define LIBINTX_ENGINE_OS_RECURRENCE_H

#include "libintx/array.h"
#include "libintx/orbital.h"
#include "libintx/recurrence.h"
#include <utility>

namespace libintx::os::vrr1 {

  inline int memory(int L) {
    int M = 0;
    for (int l = 0; l <= L; ++l) {
      M += (l+1)*ncart(L-l);
    }
    //printf("vrr1(L=%i) memory: %i\n", L, M);
    return M;
  }

  template<int L>
  auto orbital_index_sequence() {
    return std::make_index_sequence<ncart(L)>{};
  }

  struct Recurrence {
    int axis;
    int index0;
    int index1;
    int value;
  };

  inline constexpr int Axis(const Orbital &f) {
    int Axis = 0;
    if (!f[Axis] || (f[1] && f[1] < f[Axis])) Axis = 1;
    if (!f[Axis] || (f[2] && f[2] < f[Axis])) Axis = 2;
    return Axis;
  }

  // L+2 = x*(L+1) + c*(L+0)
  template<int L, size_t Idx2>
  constexpr auto make_recurrence() {
    static_assert(L > 0, "");
    constexpr auto Idx = std::integral_constant<size_t,cartesian::index(L)+Idx2>();
    constexpr auto f = cartesian::orbital(Idx);
    constexpr auto Axis = vrr1::Axis(f);
    constexpr auto Idx0 = cartesian::index(f-Orbital::Axis{Axis,2});
    constexpr auto Idx1 = cartesian::index(f-Orbital::Axis{Axis,1});
    return Recurrence{ Axis, Idx0, Idx1, f[Axis]-1 };
  }

  template<int L, int M, size_t Idx>
  void transfer1(
    const double *X0, const double *X1,
    double C0, double C1,
    const double* __restrict__ V0,
    const double* __restrict__ V1,
    double* __restrict__ A)
  {
    constexpr auto r = make_recurrence<L,Idx>();
    static const int Axis = r.axis;
    static const int factor = r.value;

    V0 += r.index0*(M+2);
    V1 += r.index1*(M+1);
    A += Idx*M;

    double x0 = X0[Axis];
    double x1 = X1[Axis];

    int m = 0;

#ifdef LIBINTX_SIMD_128
    for (; m < M-1; m += 2) {
      auto p0 = simd::load2(&V1[m+0]);
      auto p1 = simd::load2(&V1[m+1]);
      auto Am = p0*x0 - p1*x1;
      if constexpr (factor > 0) {
        auto s0 = C0*simd::load2(&V0[m+0]);
        auto s1 = C1*simd::load2(&V0[m+1]);
        Am += factor*(s0-s1);
      }
      simd::store(&A[m], Am);
    }
#endif

    for (; m < M; ++m) {
      auto p0 = (V1[m+0]);
      auto p1 = (V1[m+1]);
      auto Am = p0*x0 - p1*x1;
      if constexpr (factor > 0) {
        auto s0 = C0*(V0[m+0]);
        auto s1 = C1*(V0[m+1]);
        Am += factor*(s0-s1);
      }
      A[m] = Am;
    }

  }

  template<int L, int M, size_t ... Idx>
  void transfer_block(
    const double *X0, const double *X1,
    double C0, double C1,
    const double *V0, const double *V1,
    double *V,
    std::index_sequence<Idx...>)
  {
    (vrr1::transfer1<L,M,Idx>(X0, X1, C0, C1, V0, V1, V) , ...);
  }


  template<int L, int M>
  void transfer(
    const double* Xpa, const double* Xpq,
    double C0, double C1,
    const double *V0, const double *V1, double *V,
    std::integral_constant<int,M>)
  {
    constexpr auto orbital_index_sequence = std::make_index_sequence<ncart(L)>{};
    static_assert(M > 0);
    if constexpr (M == 1) {
      // last transfer, results got into V
      vrr1::transfer_block<L,M>(
        Xpa, Xpq, C0, C1, V0, V1,
        V+cartesian::index(L),
        orbital_index_sequence
      );
    }
    else {
      double V2[ncart(L)*M];
      vrr1::transfer_block<L,M>(Xpa, Xpq, C0, C1, V0, V1, V2, orbital_index_sequence);
      for (int i = 0; i < ncart(L); ++i) {
        V[cartesian::index(L) + i] = V2[i*M];
      }
      transfer<L+1>(
        Xpa, Xpq, C0, C1, V1, V2, V,
        std::integral_constant<int,M-1>{}
      );
    }
  }

  template<int L>
  //LIBINTX_NOINLINE
  void vrr1(
    const double* Xpa, const double* Xpq,
    double one_over_2p, double alpha_over_p,
    double *V0)
  {
    // vrr1_m populates A as [s*(L+1), p*(L), d*(L-1), ... ]
    if constexpr (L) {
      double c0 = one_over_2p;
      double c1 = c0*alpha_over_p;
      double s[L+1];
      for (int i = 0; i < L+1; ++i) {
        s[i] = V0[i];
      }
      transfer<1>(
        Xpa, Xpq, c0, c1, nullptr, s, V0,
        std::integral_constant<int,L>{}
      );
    }
  }

// 000  1.827  1.277  1.198
// 100  2.553  1.544  1.404
// 200  1.922  1.229  1.125
// 300  1.641  1.065  0.976
// 400  1.641  1.219  1.152
// 110  2.036  1.253  1.148
// 210  1.912  1.139  1.035
// 310  1.643  1.022  0.934
// 410  1.866  1.362  1.267
// 220  1.541  1.021  0.947
// 320  1.397  0.974  0.892
// 420  1.763  1.372  1.292
// 330  1.139  0.881  0.846
// 430  1.812  1.528  1.456
// 440  1.761  1.578  1.579

}

#endif /* LIBINTX_ENGINE_OS_VRR1_H */
