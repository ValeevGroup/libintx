#ifndef BOYS_ASYMPTOTIC_H
#define BOYS_ASYMPTOTIC_H

#include "libintx/forward.h"
#include <cmath>

namespace libintx::boys {

#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ __host__
#endif
  inline double asymptotic(double x, int m) {
    // cost = 1 div + 1 sqrt + (1 + 2*(m-1)) muls
    const double one_over_x = 1/x;
#ifdef __CUDACC__
    const double rsqrt_x = rsqrt(x);
#else
    const double rsqrt_x = std::sqrt(one_over_x);
#endif
    double Fm = 0.88622692545275801365 * rsqrt_x; // see Eq. (9.8.9) in Helgaker-Jorgensen-Olsen
    // this upward recursion formula omits - e^(-x)/(2x), which for x>T_crit is small enough to guarantee full double precision
    for (int i = 1; i <= m; ++i) {
      Fm = Fm * (i - 0.5) * one_over_x; // see Eq. (9.8.13)
    }
    return Fm;
  }

  template<typename T, int N>
#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ __host__
#endif
  LIBINTX_ALWAYS_INLINE
  void asymptotic_1_x(T one_over_x, int m, T (&F)[N]) {
    // cost = 1 div + 1 sqrt + (1 + 2*(m-1)) muls
    using std::sqrt;
    T Fm = 0.88622692545275801365 * sqrt(one_over_x); // see Eq. (9.8.9) in Helgaker-Jorgensen-Olsen
    // this upward recursion formula omits - e^(-x)/(2x), which for x>T_crit is small enough to guarantee full double precision
    for (int i = 1; i <= m; ++i) {
      Fm = Fm * (i - 0.5) * one_over_x; // see Eq. (9.8.13)
    }
    F[0] = Fm;
    for (int i = 1; i < N; ++i) {
      Fm = Fm * (m+i - 0.5) * one_over_x;
      F[i] = Fm;
    }
  }

  template<typename T, int N>
#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ __host__
#endif
  LIBINTX_ALWAYS_INLINE
  void asymptotic(T x, int m, T (&F)[N]) {
    asymptotic_1_x(1/x, m, F);
  }

}

#endif /* BOYS_ASYMPTOTIC_H */
