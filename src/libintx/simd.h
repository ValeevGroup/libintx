#ifndef LIBINTX_SIMD_H
#define LIBINTX_SIMD_H

#include "libintx/forward.h"
#include "libintx/config.h"

#ifdef LIBINTX_SIMD

#if __has_include(<simd>)
#include <simd>
#elif __has_include(<experimental/simd>)
#include <experimental/simd>
#else
#error "No std::simd library"
#endif

#else

#pragma message("libintx::simd disabled")

#endif

namespace libintx::simd {

#ifdef LIBINTX_SIMD

  template<typename T>
  using simd_t = std::experimental::native_simd<T>;

#define LIBINTX_SIMD_DOUBLE libintx::simd::simd_t<double>

  using std::experimental::is_simd_v;

#else

  template<typename T>
  using simd_t = T;

  template<typename T>
  constexpr auto is_simd_v = std::false_type{};

#endif

  template<typename T, size_t N = 1>
  constexpr size_t size = [](){
    if constexpr (is_simd_v<T>) {
      return T::size();
    }
    else return N;
  }();

}

namespace libintx {
  using simd::simd_t;
  using simd::is_simd_v;
}

#if defined(LIBINTX_SIMD)
#if defined(__AVX512F__)
#pragma message("Using AVX512 vectorisation")
static_assert(sizeof(libintx::simd_t<double>) == 64);
#define LIBINTX_SIMD_ISA "AVX512"
#elif defined(__AVX__)
#pragma message("Using AVX vectorisation")
static_assert(sizeof(libintx::simd_t<double>) == 32);
#define LIBINTX_SIMD_ISA "AVX"
#elif defined(__SSE__)
#pragma message("Using SSE vectorisation")
static_assert(sizeof(libintx::simd_t<double>) == 16);
#define LIBINTX_SIMD_ISA "SSE"
#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
#pragma message("Using NEON vectorisation")
static_assert(sizeof(libintx::simd_t<double>) == 16);
#define LIBINTX_SIMD_ISA "NEON"
#endif
#endif // LIBINTX_SIMD

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define LIBINTX_SIMD_REGISTERS 32
#else
#define LIBINTX_SIMD_REGISTERS 16
#endif

#endif /* LIBINTX_SIMD_H */
