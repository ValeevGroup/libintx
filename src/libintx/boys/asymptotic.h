#ifndef BOYS_ASYMPTOTIC_H
#define BOYS_ASYMPTOTIC_H

#include <cmath>

namespace boys {

#ifdef __CUDACC__
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

  template<int N>
#ifdef __CUDACC__
  __device__ __host__
#endif
  inline void asymptotic(double x, int m, double (&F)[N]) {
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
    F[0] = Fm;
    for (int i = 1; i < N; ++i) {
      Fm = Fm * (m+i - 0.5) * one_over_x;
      F[i] = Fm;
    }
  }

}

#endif /* BOYS_ASYMPTOTIC_H */
