#ifndef LIBINTX_ENGINE_OS_VRR2_H
#define LIBINTX_ENGINE_OS_VRR2_H

#include "libintx/array.h"
#include "libintx/orbital.h"
#include "libintx/recurrence.h"
#include "libintx/simd.h"
#include <utility>

namespace libintx::os::vrr2 {

  using cartesian::orbital_list;
  using recurrence::recurrence_table;

  template<int AB, int X>
  //LIBINTX_NOINLINE
  inline void evaluate_terms(
    int A, int B, std::integral_constant<int,X>,
    int K, const double *Xpc, const double *alpha_over_2pc,
    double *V1, double *V2,
    std::integral_constant<int,X>)
  {
    if (X > AB) return;

    const int A0 = std::max(A-X,0);
    const int NAB = ncartsum(AB)-ncartsum(A-1);

    // optimization trick, pre-contract V1
    simd::scale(ncartsum(AB-X), alpha_over_2pc[0], V1);
    for (int k = 1; k < K; ++k) {
      const auto *Vk =  V1+k*ncartsum(AB);
      simd::axpy(ncartsum(AB-X), alpha_over_2pc[k], Vk, V1);
    }

    for (int i = cartesian::index(A0); i < cartesian::index(AB-X+1); ++i) {
      double v1 = V1[i];
      auto *v2 = V2;
      for (int j = cartesian::index(X); j < cartesian::index(X+1); ++j) {
        const auto &r = recurrence::recurrence_table[j][i];
        double c = r.coefficient;
        v2[r.index] += c*v1;
        v2 += NAB;
      }
    }

  }

  template<int AB, int X, int T>
  //LIBINTX_NOINLINE
  inline void evaluate_terms(
    int A, int B, std::integral_constant<int,X>,
    int K, const double *Xpc, const double *alpha_over_2pc,
    double *V1, double *V2,
    std::integral_constant<int,T>)
  {

    if (T > AB) return;

    const int A0 = std::max(0, A-T);
    const int NAB = ncartsum(AB)-ncartsum(A-1);

    // optimization trick, pre-scale V1
    for (int k = 0; k < K; ++k) {
      auto *Vk = V1+k*ncartsum(AB);
      simd::scale(ncartsum(AB-T), alpha_over_2pc[k], Vk);
    }

    for (int xt = cartesian::index(X-T); xt < cartesian::index(X-T+1); ++xt) {
      double V[std::max(0,ncartsum(AB-T))] = { };
      for (int k = 0; k < K; ++k) {
        const auto *Vk = V1+k*ncartsum(AB);
        double r = pow(Xpc+k*3, orbital_list[xt]);
        simd::axpy(
          ncartsum(AB-T)-cartesian::index(A0),
          r,
          Vk + cartesian::index(A0),
          V + cartesian::index(A0)
        );
      }
      for (int t = cartesian::index(T); t < cartesian::index(T+1); ++t) {
        auto x = recurrence_table[xt][t];
        x.index -= cartesian::index(X);
        double m = x.multiplicity;
        auto* Vx = V2 + x.index*NAB;
        for (int i = cartesian::index(A0); i < cartesian::index(AB+1-T); ++i) {
          auto a = recurrence_table[t][i];
          double c = m*a.coefficient;
          Vx[a.index] += c*V[i];
        }
      }
    }

    evaluate_terms<AB>(
      A, B, std::integral_constant<int,X>{},
      K, Xpc, alpha_over_2pc, V1, V2,
      std::integral_constant<int,T+1>{}
    );

  }

  // Term-0 + rest of terms
  template<int AB, int X>
  void evaluate_terms(
    int A, int B, std::integral_constant<int,X>,
    int K, const double *Xpc, const double *alpha_over_2pc,
    double *V1, double *V2)
  {
    const int NAB = ncartsum(AB)-ncartsum(A-1);
    for (int x = 0; x < ncart(X); ++x) {
      auto *Vx = V2 + x*NAB;
      for (int k = 0; k < K; ++k) {
        double r = pow(Xpc+k*3, orbital_list[cartesian::index(X)+x]);
        const auto* Vk = V1+k*ncartsum(AB);
        simd::axpy(NAB, r, Vk+cartesian::index(A), Vx+cartesian::index(A));
      }
    }
    evaluate_terms<AB>(
      A, B, std::integral_constant<int,X>{},
      K, Xpc, alpha_over_2pc, V1, V2,
      std::integral_constant<int,1>{}
    );
  }

  template<int AB, int X>
  void vrr2(
    int A, int B, std::integral_constant<int,X>,
    int K, const double *Xpc, const double *alpha_over_2pc,
    double *V1, double *V2)
  {
    V2 = V2 - cartesian::index(A); // re-position relative to L=0
    evaluate_terms<AB>(
      A, B, std::integral_constant<int,X>{},
      K, Xpc, alpha_over_2pc, V1, V2
    );
  }

  template<int AB>
  inline void vrr2(
    int A, int B, std::integral_constant<int,0>,
    int K, const double *Xpc, const double *alpha_over_2pc,
    double *V1, double *V2)
  {
    const double *Vk = V1 + cartesian::index(A);
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < ncartsum(AB); ++i) {
        V2[i] += Vk[i];
      }
      Vk += ncartsum(AB);
    }
  }

}

#endif /* LIBINTX_ENGINE_OS_VRR2_H */
