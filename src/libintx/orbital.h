#ifndef LIBINTX_ORBITAL_H
#define LIBINTX_ORBITAL_H

#include "libintx/forward.h"

#include "libintx/array.h"
#include "libintx/math.h"
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

template<typename ... Args>
LIBINTX_GPU_ENABLED
constexpr inline int ncart(int L, Args ...  Ls) {
  return ncart(L)*(ncart(Ls) * ...);
}

LIBINTX_GPU_ENABLED
constexpr inline int ncartsum(int L) {
  //assert(L >= -1);
  return ((L+1)*(L+2)*(L+3))/6;
}

template<typename Index = int>
LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr int index(Index i, Index j, Index k) {
  const auto jk = j+k;
  return (jk*(jk+1))/2 + k;
}

template<typename Index>
LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr int index(Index L) {
  return (L*(L+1)*(L+2))/6;
}

// orbital index relative to shell L
template<int L, typename Index = int>
LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr int index(Index i, Index j, Index k) {
  return (index(i+j+k) + index(i,j,k) - index(L));
}

struct alignas(4) Orbital {

  constexpr static Orbital Axis(int axis, uint8_t value = 1) {
    return {
      axis == 0 ? value : uint8_t(0),
      axis == 1 ? value : uint8_t(0),
      axis == 2 ? value : uint8_t(0)
    };
  };

  uint8_t lmn[3] = { 0xFF, 0xFF, 0xFF };

  LIBINTX_GPU_ENABLED
  constexpr auto L() const {
    auto [l,m,n] = lmn;
    return (l+m+n);
  }

  template<size_t Idx>
  LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
  constexpr auto get() const {
    return lmn[Idx];
  }

  LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
  constexpr auto& operator[](int k) {
    return lmn[k];
  }

  LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
  constexpr const auto& operator[](int k) const {
    return lmn[k];
  }

};

LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr int index(const Orbital &t) {
  auto [i,j,k] = t;
  return cartesian::index(i,j,k);
}

template<int L>
LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr auto index(const Orbital &t) {
  const auto [i,j,k] = t;
  return cartesian::index<L>(i,j,k);
}

LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr Orbital operator+(const Orbital &p, const Orbital &q) {
  return Orbital{{
    uint8_t(p[0] + q[0]),
    uint8_t(p[1] + q[1]),
    uint8_t(p[2] + q[2]),
  }};
}

LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr Orbital operator-(const Orbital &p, const Orbital &q) {
  return Orbital{{
    uint8_t(p[0] - q[0]),
    uint8_t(p[1] - q[1]),
    uint8_t(p[2] - q[2]),
  }};
}

LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr bool operator<=(const Orbital &p, const Orbital &q) {
  return ((p[0] <= q[0]) && (p[1] <= q[1]) && (p[2] <= q[2]));
}

template<typename Orbital = libintx::cartesian::Orbital, size_t ... Ls>
constexpr auto make_orbitals(std::index_sequence<Ls...>) {
  constexpr int N = (ncart(Ls) + ...);
  std::array<Orbital,N> orbitals = {};
  size_t idx = 0;
  for (int l : { Ls...}) {
    for (uint8_t k = 0; k <= l; ++k) {
      for (uint8_t j = 0; j <= l-k; ++j) {
        uint8_t i = l - (j+k);
        auto orbital = Orbital{{i,j,k}};
        orbitals[idx+index(orbital)] = orbital;
      }
    }
    idx += ncart(l);
  }
  return orbitals;
}

template<int L, typename T = Orbital>
constexpr auto orbitals() {
  return make_orbitals<T>(std::index_sequence<L>());
}

template<int L = 0, size_t Idx>
constexpr Orbital orbital(std::integral_constant<size_t,Idx> = {}) {
  for (int l = L; ; ++l) {
    for (uint8_t k = 0; k <= l; ++k) {
      for (uint8_t j = 0; j <= l-k; ++j) {
        uint8_t i = l - (j+k);
        auto orbital = Orbital{{i,j,k}};
        if (index<L>(orbital) == Idx) return orbital;
      }
    }
  }
}

//LIBINTX_GPU_ENABLED
inline double pow(const double *r3, const Orbital &p) {
  double r = 1;
  for (int i = 0; i < 3; ++i) {
    r *= math::ipow(r3[i], p[i]);
  }
  return r;
}

LIBINTX_GPU_CONSTANT
constexpr auto orbitals2 = make_orbitals(
  std::make_index_sequence<std::max(XMAX,2*LMAX)+1>()
);

LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr const Orbital* begin(int L) {
  assert(index(L+1) <= (int)orbitals2.size());
  return &orbitals2[0]+index(L);
}

LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr const Orbital* end(int L) {
  assert(index(L+1) <= (int)orbitals2.size());
  return &orbitals2[0]+index(L+1);
}

LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr const Orbital& orbital(int L, int idx) {
  return begin(L)[idx];
}

template<typename T = Orbital>
constexpr auto orbitals(int L) {
  return range<const T*>(begin(L), end(L));
}

}

namespace libintx::pure {

struct Orbital {
  int16_t l,m;
};

LIBINTX_GPU_ENABLED
constexpr inline Orbital orbital(int L, int idx) {
  return Orbital{ int16_t(L), int16_t(idx-L) };
}

LIBINTX_GPU_ENABLED
constexpr inline int npure(int L) {
  return (L*2+1);
}

template<typename ... Args>
LIBINTX_GPU_ENABLED
constexpr inline int npure(int L, Args ...  Ls) {
  return npure(L)*(npure(Ls) * ...);
}

constexpr inline int index(int L) {
  return (L*L);
}

constexpr inline int index(int l, int m) {
  return (l+m);
}

constexpr inline int index(const Orbital &o) {
  return index(o.l,o.m);
}

template<int L>
constexpr auto orbitals() {
  array<Orbital,npure(L)> s = {};
  for (size_t i = 0; i < s.size(); ++i) {
    s[i] = orbital(L,i);
  }
  return s;
}

}

namespace libintx::hermite {

LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr int nherm2(int L) {
  return cartesian::ncartsum(L);
}

LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr int nherm1(int L) {
  int n = 0;
  for (int i = L; i >= 0; i -= 2) {
    n += cartesian::ncart(i);
  }
  return n;
}

template<class Orbital>
LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr auto index2(const Orbital &h) {
  return cartesian::index<0>(h);
}

template<typename Tx, typename Ty, typename Tz>
LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr auto index1(const Tx &lx, const Ty &ly, const Tz &lz) {
  int L = lx + ly + lz;
  int idx = 0;
  for (int i = L%2; i < L; i += 2) {
    idx += cartesian::ncart(i);
  }
  return (idx + cartesian::index(lx,ly,lz));
}

template<class Orbital>
LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
constexpr auto index1(const Orbital &h) {
  return index1(h[0], h[1], h[2]);
}

template<int L>
LIBINTX_GPU_CONSTANT
constexpr auto orbitals2 = cartesian::make_orbitals(
  std::make_index_sequence<L+1>()
);

template<int L, int Parity>
LIBINTX_GPU_CONSTANT
constexpr auto orbitals1 = []() {
  constexpr int N = nherm1(L%2 == Parity ? L : L-1);
  array<cartesian::Orbital,N> s = {};
  int idx = 0;
  for (auto &t : orbitals2<L>) {
    if (t.L()%2 != Parity) continue;
    s[idx++] = t;
  }
  assert(idx == N);
  return s;
 }();

template<class Orbital>
inline constexpr auto phase(const Orbital &p) {
  return (p.L()%2 ? -1 : +1);
}

} // libintx::hermite

namespace libintx {
  using libintx::cartesian::ncart;
  using libintx::cartesian::ncartsum;
  using libintx::cartesian::Orbital;
  using libintx::pure::npure;
  using libintx::hermite::nherm1;
  using libintx::hermite::nherm2;
}

#endif /* LIBINTX_SHELL_H */
