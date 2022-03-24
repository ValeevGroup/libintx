#ifdef RYSQ_VECTORIZE

#if defined(__AVX__)
#warning Using AVX
#define RYSQ_VECTORIZE_AVX
#elif defined(__SSE__)
#warning Using SSE
#define RYSQ_VECTORIZE_SSE
#endif

#endif

#define RYSQ_SIMD_INTRINSIC(NAME, INTRINSIC)      \
  template<typename ... Args>                     \
  inline auto NAME(const Args & ... args) {       \
    return INTRINSIC(args...);                    \
  }

#ifdef RYSQ_VECTORIZE_AVX
#define RYSQ_VECTORIZE

#include <immintrin.h>

namespace rysq {
namespace simd {

  constexpr int size() {
    return 4;
  }

  RYSQ_SIMD_INTRINSIC(set1, _mm256_set1_pd);
  RYSQ_SIMD_INTRINSIC(zero, _mm256_setzero_pd);
  RYSQ_SIMD_INTRINSIC(load, _mm256_load_pd);
  RYSQ_SIMD_INTRINSIC(store, _mm256_storeu_pd);

  inline double hadd(__m256d r) {
    __m256d t1 = _mm256_hadd_pd(r,r);
    __m128d t2 = _mm256_extractf128_pd(t1,1);
    __m128d t3 = _mm_add_sd(_mm256_castpd256_pd128(t1),t2);
    return _mm_cvtsd_f64(t3);
  }

}
}

#endif

#ifdef RYSQ_VECTORIZE_SSE
#define RYSQ_VECTORIZE

#include <pmmintrin.h>

namespace rysq {
namespace simd {

  constexpr int size() {
    return 2;
  }

  RYSQ_SIMD_INTRINSIC(set1, _mm_set1_pd);
  RYSQ_SIMD_INTRINSIC(zero, _mm_setzero_pd);
  RYSQ_SIMD_INTRINSIC(load, _mm_load_pd);
  RYSQ_SIMD_INTRINSIC(store, _mm_storeu_pd);

  inline double hadd(__m128d r) {
    r = _mm_hadd_pd(r,r);
    return r[0];
  }

}
}

#endif
