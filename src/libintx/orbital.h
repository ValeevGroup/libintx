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

namespace libintx {

  template<typename It>
  struct orbital_range {
    constexpr orbital_range(It begin, It end)
      : begin_(begin), end_(end) {}
    constexpr It begin() const { return begin_; }
    constexpr It end() const { return end_; }
  private:
    It begin_, end_;
  };

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

LIBINTX_GPU_ENABLED
constexpr inline int ncart(int M, int L) {
  return ncartsum(L) - (M ? ncartsum(M-1) : 0);
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
  uint8_t _L = 0xFF;

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
    uint8_t(p[2] + q[2]),
    uint8_t(p._L + q._L)
  };
}

LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr Orbital operator-(const Orbital &p, const Orbital &q) {
  return Orbital{
    uint8_t(p[0] - q[0]),
    uint8_t(p[1] - q[1]),
    uint8_t(p[2] - q[2]),
    uint8_t(p._L - q._L)
  };
}

LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr bool operator<=(const Orbital &p, const Orbital &q) {
  return ((p[0] <= q[0]) && (p[1] <= q[1]) && (p[2] <= q[2]));
}

template<int L, typename T = Orbital>
constexpr auto shell() {
  std::array<T,ncart(L)> orbitals = {};
  for (uint8_t k = 0; k <= L; ++k) {
    for (uint8_t j = 0; j <= L-k; ++j) {
      uint8_t i = L - (j+k);
      //printf("%i,%i,%i\n", i, j, k);
      orbitals[index(i,j,k)] = T{{i,j,k},L};
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
constexpr auto index_sequence_ab = index_sequence<std::max(XMAX,2*LMAX)>;

LIBINTX_GPU_CONSTANT
constexpr auto orbital_list = make_array<Orbital>(
  [](auto idx) { return orbital(idx); },
  index_sequence_ab
);

LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr const Orbital* begin(int L) {
  assert(index(L) < (int)orbital_list.size());
  return &orbital_list[0]+index(L);
}

LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr const Orbital* end(int L) {
  assert(index(L) < (int)orbital_list.size());
  return &orbital_list[0]+index(L+1);
}

LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr const Orbital& orbital(int L, int idx) {
  return begin(L)[idx];
}

template<int L>
LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr const Orbital& orbital(int idx) {
  return begin(L)[idx];
}

LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr const Orbital& orbital(int idx) {
  assert(idx < (int)orbital_list.size());
  return orbital_list[idx];
}

template<typename T = Orbital>
constexpr auto shell(int L) {
  return orbital_range<const T*>(
    begin(L), end(L)
  );
}

// template<typename T = Orbital>
// constexpr auto range(int first, int last) {
//   return orbital_range<const T*>(
//     begin(first), begin(first)+last
//   );
// }

// template<typename T = Orbital>
// constexpr auto range(int last) {
//   return range(0,last);
// }

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
constexpr auto shell() {
  array<Orbital,npure(L)> s = {};
  for (int i = 0; i < s.size(); ++i) {
    s[i] = orbital(L,i);
  }
  return s;
}

}

namespace libintx::hermite {

LIBINTX_GPU_ENABLED
constexpr inline int nherm2(int L) {
  return cartesian::ncartsum(L);
}

LIBINTX_GPU_ENABLED
constexpr inline int nherm1(int L) {
  int n = 0;
  for (int i = L; i >= 0; i -= 2) {
    n += cartesian::ncart(i);
  }
  return n;
}

template<class Orbital>
LIBINTX_GPU_ENABLED
constexpr inline auto index2(const Orbital &h) {
  return cartesian::index<0>(h);
}

template<typename Tx, typename Ty, typename Tz>
LIBINTX_GPU_ENABLED
constexpr auto index1(const Tx &lx, const Ty &ly, const Tz &lz) {
  int L = lx + ly + lz;
  int idx = 0;
  for (int i = L%2; i < L; i += 2) {
    idx += cartesian::ncart(i);
  }
  return (idx + cartesian::index(lx,ly,lz));
}

template<class Orbital>
LIBINTX_GPU_ENABLED
constexpr auto index1(const Orbital &h) {
  return index1(h[0], h[1], h[2]);
}

// LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
// constexpr auto orbitals(int L) {
//   return orbital_range(cartesian::begin(0), cartesian::end(L));
// }

template<int L>
LIBINTX_GPU_CONSTANT
constexpr auto orbitals = cartesian::orbitals(cartesian::index_sequence<L>);

template<int L, int Parity>
LIBINTX_GPU_CONSTANT
constexpr auto orbitals1 = []() {
  constexpr int N = nherm1(L%2 == Parity ? L : L-1);
  array<cartesian::Orbital,N> s = {};
  int idx = 0;
  for (auto &t : orbitals<L>) {
    if (t.L()%2 != Parity) continue;
    s[idx++] = t;
  }
  assert(idx == N);
  return s;
 }();

LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
constexpr const auto& orbital(int idx) {
  return orbitals<std::max(XMAX+2*LMAX,4*LMAX)>[idx];
}

}

namespace libintx {
  using libintx::cartesian::ncart;
  using libintx::cartesian::ncartsum;
  using libintx::cartesian::Orbital;
  using libintx::pure::npure;
  using libintx::hermite::nherm1;
  using libintx::hermite::nherm2;
}

#endif /* LIBINTX_SHELL_H */
