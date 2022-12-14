#ifndef LIBINTX_ORBITAL_H
#define LIBINTX_ORBITAL_H

#include "libintx/array.h"
#include "libintx/math.h"
#include "libintx/forward.h"
#include "libintx/config.h"

#include <cassert>
#include <cstdint>
#include <tuple>
//#include <strings.h> // ffs

namespace libintx::cartesian {

  struct Orbital;

}

namespace std {

  template<>
  struct tuple_size<libintx::cartesian::Orbital>
    : std::integral_constant<size_t,3> {};

  template<std::size_t I>
  struct tuple_element<I, libintx::cartesian::Orbital>
  {
     using type = uint8_t;
  };

}

namespace libintx::cartesian {

LIBINTX_GPU_ENABLED
constexpr inline int ncart(int L) {
  return ((L+1)*(L+2))/2;
}

LIBINTX_GPU_ENABLED
constexpr inline int ncartsum(int L) {
  //assert(L >= -1);
  return ((L+1)*(L+2)*(L+3))/6;
}

template<typename Index = int>
LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr int index(Index i, Index j, Index k) {
  const auto jk = j+k;
  return (jk*(jk+1))/2 + k;
}

template<typename Index>
LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr int index(Index L) {
  return (L*(L+1)*(L+2))/6;
}

// orbital index relative to shell L
template<int L, typename Index = int>
LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr int index(Index i, Index j, Index k) {
  return (index(i+j+k) + index(i,j,k) - index(L));
}

struct Orbital {

  struct Axis {
    int axis;
    int value = 1;
    LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
    constexpr operator int() const {
      return axis;
    }
  };

  uint8_t lmn[3] = { 0xFF, 0xFF, 0xFF };

  LIBINTX_GPU_ENABLED
  constexpr auto L() const {
    auto [l,m,n] = lmn;
    return (l+m+n);
  }

  template<size_t Idx>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  constexpr auto get() const {
    return lmn[Idx];
  }

  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  constexpr auto& operator[](int k) {
    return lmn[k];
  }

  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  constexpr const auto& operator[](int k) const {
    return lmn[k];
  }

  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  constexpr Orbital operator-(const Axis &x) const {
    Orbital p = *this;
    p.lmn[x] -= x.value;
    return p;
  }

  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  constexpr Orbital operator+(const Axis &x) const {
    Orbital p = *this;
    p.lmn[x] += x.value;
    return p;
  }

};

LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr int index(const Orbital &t) {
  auto [i,j,k] = t;
  return cartesian::index(i,j,k);
}

template<int L>
LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr auto index(const Orbital &t) {
  const auto [i,j,k] = t;
  return cartesian::index<L>(i,j,k);
}

inline constexpr uint16_t bitstring(const Orbital &f) {
  constexpr uint16_t y = 0x5555; // 0101...
  constexpr uint16_t z = 0xAAAA; // 1010...
  return (
    // x are all 0
    (y & ((1 << 2*(uint16_t)f[1]) - 1)) << 2*(f[0]) |
    (z & ((1 << 2*(uint16_t)f[2]) - 1)) << 2*(f[0]+f[1])
  );
}


// LIBINTX_GPU_ENABLED
// inline int ffs(const Orbital &o) {
// #ifdef __CUDA_ARCH__
//   return __ffs(o.data)/8;
// #else
//   return ::ffs(o.data);
// #endif
// }

LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr Orbital operator+(const Orbital &p, const Orbital &q) {
  return Orbital{
    uint8_t(p[0] + q[0]),
    uint8_t(p[1] + q[1]),
    uint8_t(p[2] + q[2])
  };
}

LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr Orbital operator-(const Orbital &p, const Orbital &q) {
  return Orbital{
    uint8_t(p[0] - q[0]),
    uint8_t(p[1] - q[1]),
    uint8_t(p[2] - q[2])
  };
}

LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr bool operator<=(const Orbital &p, const Orbital &q) {
  return ((p[0] <= q[0]) && (p[1] <= q[1]) && (p[2] <= q[2]));
}

template<int L, typename T = Orbital>
constexpr auto shell() {
  array<T,ncart(L)> orbitals = {};
  for (uint8_t k = 0; k <= L; ++k) {
    for (uint8_t j = 0; j <= L-k; ++j) {
      uint8_t i = L - (j+k);
      //printf("%i,%i,%i\n", i, j, k);
      orbitals[index(i,j,k)] = T{i,j,k};
    }
  }
  return orbitals;
}

template<int L = 0, size_t Idx>
constexpr Orbital orbital(std::integral_constant<size_t,Idx> = {}) {
  if constexpr ((int)Idx >= ncart(L)) {
    return orbital<L+1,Idx-ncart(L)>();
  }
  return shell<L>()[Idx];
}

template<size_t ... Idx>
constexpr auto orbitals(std::index_sequence<Idx...> = {}) {
  return array{ orbital<0,Idx>()... };
}

  //LIBINTX_GPU_ENABLED
inline double pow(const double *r3, const Orbital &p) {
  double r = 1;
  for (int i = 0; i < 3; ++i) {
    r *= math::ipow(r3[i], p[i]);
  }
  return r;
}

template<int L>
constexpr auto index_sequence = std::make_index_sequence<ncartsum(L)>();

constexpr auto index_sequence_x = index_sequence<XMAX>;
constexpr auto index_sequence_ab = index_sequence<2*LMAX>;

LIBINTX_GPU_CONSTANT
const auto orbital_list = make_array<Orbital>(
  [](auto idx) { return orbital(idx); },
  index_sequence_ab
);

LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr const Orbital* begin(int L) {
  return &orbital_list[0]+index(L);
}

LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr const Orbital* end(int L) {
  return &orbital_list[0]+index(L+1);
}

LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr const Orbital& orbital(int L, int idx) {
  return begin(L)[idx];
}

}

namespace libintx::pure {

struct Orbital {
  int16_t l,m;
};

LIBINTX_GPU_ENABLED
constexpr inline int npure(int L) {
  return (L*2+1);
}

constexpr inline size_t index(int L) {
  return (L*L);
}

constexpr inline size_t index(int l, int m) {
  return (l+m);
}

constexpr inline size_t index(const Orbital &o) {
  return index(o.l,o.m);
}

}

namespace libintx {
  using libintx::cartesian::ncart;
  using libintx::cartesian::ncartsum;
  using libintx::cartesian::Orbital;
  using libintx::pure::npure;
}

#endif /* LIBINTX_SHELL_H */
