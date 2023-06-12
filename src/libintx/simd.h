#ifndef LIBINTX_SIMD_H
#define LIBINTX_SIMD_H

#include "libintx/forward.h"
#include "libintx/config.h"

#ifdef __AVX__
#include <immintrin.h>
#define LIBINTX_SIMD_AVX
#endif

#ifdef __SSE3__
#include <xmmintrin.h>
#define LIBINTX_SIMD_128
#endif

namespace libintx {
namespace simd {

  template<size_t K = 1>
  inline void transpose(size_t m, size_t n, const double *A, double *B) {
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        int ij = i+j*m;
        int ji = j+i*n;
        for (size_t k = 0; k < K; ++k) {
          B[k+ji*K] = A[k+ij*K];
        }
      }
    }
  }

  template<typename T>
  inline void copy(size_t N, const T *src, T *dst) {
    for (size_t i = 0; i < N; ++i) {
      dst[i] = src[i];
    }
  }

#ifdef LIBINTX_SIMD_128

  inline __m128d load2(const double* p) {
    return _mm_loadu_pd(p);
  }

  inline void store(double* p, __m128d v) {
    _mm_storeu_pd(p, v);
  }

#endif

  LIBINTX_NOINLINE
  inline void scale(int N, double s, double* __restrict__ V) {
    int i = 0;
#ifdef LIBINTX_SIMD_AVX
    for (; i < N-3; i += 4) {
      auto v = _mm256_loadu_pd(V+i);
      v *= s;
      _mm256_storeu_pd(V+i, v);
    }
#endif
    for (; i < N; ++i) {
      V[i] *= s;
    }
  }

  LIBINTX_NOINLINE
  inline void axpy(int N, double A, const double* __restrict__ U, double* __restrict__ V) {
    int i = 0;
#ifdef LIBINTX_SIMD_AVX
    for (; i < N-3; i += 4) {
      auto u = _mm256_loadu_pd(U+i);
      auto v = _mm256_loadu_pd(V+i);
      v += A*u;
      _mm256_storeu_pd(V+i, v);
    }
#endif
    for (; i < N; ++i) {
      V[i] += A*U[i];
    }
  }

  inline void multiply_add_store(
    int N, double s,
    const double * __restrict__ U1,
    const double * __restrict__ U2,
    double * __restrict__ V)
  {
    int i = 0;
#ifdef LIBINTX_SIMD_AVX
    for (; i < N-3; i += 4) {
      auto u1 = _mm256_loadu_pd(U1+i);
      auto u2 = _mm256_loadu_pd(U2+i);
      auto v = s*u1 + u2;
      _mm256_storeu_pd(V+i, v);
    }
#endif
    for (; i < N; ++i) {
      V[i] = s*U1[i] + U2[i];
    }
  }

}
}

#endif /* LIBINTX_SIMD_H */
