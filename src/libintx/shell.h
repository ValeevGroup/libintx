#ifndef LIBINTX_SHELL_H
#define LIBINTX_SHELL_H

#include "libintx/orbital.h"
#include "libintx/array.h"
#include "libintx/math.h"
#include "libintx/config.h"

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <strings.h> // ffs
#include <vector>
#include <stdexcept>

namespace libintx {

struct Shell {
  //using Orbital = shell::Orbital;
  int L;
  int pure = 1;
};

inline bool operator==(const Shell &lhs, const Shell &rhs) {
  return (
    (lhs.L == rhs.L) &&
    (lhs.pure == rhs.pure)
  );
}


LIBINTX_GPU_ENABLED
constexpr inline int ncart(const Shell &s) {
  return ncart(s.L);
}

LIBINTX_GPU_ENABLED
constexpr inline int npure(const Shell &s) {
  return npure(s.L);
}

LIBINTX_GPU_ENABLED
constexpr inline int nbf(const Shell &s) {
  return (s.pure ? npure(s) : ncart(s));
}

struct alignas(32) Gaussian : Shell {

  constexpr static int KMAX = libintx::KMAX;

  struct Primitive {
    double a = 1, C = NAN;
    template<int N>
    static auto array(std::vector<Primitive> v) {
      assert(v.size() <= N);
      libintx::array<Primitive,N> primitives;
      for (size_t k = 0; k < v.size(); ++k) {
        primitives[k] = v[k];
      }
      return primitives;
    }
  };

  array<Primitive,KMAX> prims = { };
  int K = 0;

  //LIBINTX_GPU_ENABLED
  Gaussian() = default;

  Gaussian(int L, std::vector<Primitive> ps, bool pure = true)
    : Shell({L,(int)pure}), K(ps.size())
  {
    if (ps.size() > KMAX) {
      throw std::domain_error("libintx::Gaussian: K > KMAX");
    }
    std::copy(ps.begin(), ps.end(), prims.data);
  }

};

static_assert(sizeof(Gaussian) == 192);

inline Gaussian::Primitive normalized(int L, Gaussian::Primitive p) {
  using math::factorial2_Kminus1;
  using math::sqrt_pi3;
  using math::pow;
  using math::sqrt;
  assert(L <= 15); // due to df_Kminus1[] a 64-bit integer type; kinda ridiculous restriction anyway
  auto alpha = p.a;
  assert(alpha >= 0);
  if (alpha == 0) return p;
  const auto two_alpha = 2*alpha;
  const auto two_alpha_to_am32 = pow(two_alpha,L+1) * sqrt(two_alpha);
  const auto f = sqrt(
    pow(2,L) * two_alpha_to_am32/
    (sqrt_pi3 * factorial2_Kminus1[2*L] )
  );
  double C = f*p.C;
  return Gaussian::Primitive{alpha, C };
}

inline Gaussian normalized(const Gaussian &s) {
  std::vector<Gaussian::Primitive> prims(s.K);
  for (int i = 0; i < s.K; ++i) {
    prims[i] = normalized(s.L, s.prims[i]);
  }
  return Gaussian(s.L, prims, s.pure);
}

inline auto normalization_factor(const Gaussian &g) {
  using math::factorial2_Kminus1;
  using math::sqrt_pi3;
  using std::sqrt;
  using std::pow;
  double norm = 0;
  int L = g.L;
  for (int i = 0; i < g.K; ++i) {
    for (int j = 0; j < g.K; ++j) {
      auto p = g.prims[i];
      auto q = g.prims[j];
      auto gamma = p.a + q.a;
      auto C = p.C*q.C;
      norm += C/pow(gamma,L+1.5);
    }
  }
  norm *= sqrt_pi3;
  norm *= factorial2_Kminus1[2*L];
  norm /= pow(2.0,L);
  return 1/sqrt(norm);
}


inline bool operator==(const Gaussian::Primitive &lhs, const Gaussian::Primitive &rhs) {
  return (lhs.a == rhs.a && lhs.C == rhs.C);
}

inline bool operator==(const Gaussian &lhs, const Gaussian &rhs) {
  if (!((const Shell&)lhs == (const Shell&)rhs)) return false;
  if (lhs.K != rhs.K) return false;
  for (int k = 0; k < lhs.K; ++k) {
    if (!(lhs.prims[k] == rhs.prims[k])) return false;
  }
  return true;
}

struct Gaussian2 {
  Gaussian first, second;
  struct {
    array<double,3> first, second;
  } r;
};

template<typename Shell>
using Basis = std::vector< std::tuple<Shell,array<double,3> > >;

template<typename Shell>
LIBINTX_GPU_ENABLED
constexpr auto& shell(const std::tuple< Shell,array<double,3> > &g) {
  return std::get<0>(g);
}

LIBINTX_GPU_ENABLED
inline constexpr auto& exp(const std::tuple< Gaussian, array<double,3> > &g, int k) {
  return shell(g).prims[k].a;
}

template<typename Shell>
LIBINTX_GPU_ENABLED
constexpr auto& center(const std::tuple< Shell,array<double,3> > &g) {
  return std::get<1>(g);
}

template<typename Shell>
int nbf(const std::tuple< Shell,array<double,3> > &g) {
  return nbf(std::get<0>(g));
}

template<typename Shell>
size_t nbf(Basis<Shell> &basis) {
  size_t n = 0;
  for (auto &[a,r] : basis) {
    n += nbf(a);
  }
  return n;
}


template<typename S>
struct Unit : libintx::Shell {
  using Primitive = typename S::Primitive;
  static constexpr array<Primitive,1> prims = {{ 0, 1 }};
  static constexpr int K = 1;
  constexpr Unit() : libintx::Shell({0,true}) {}
};

template<typename Shell>
LIBINTX_GPU_ENABLED
constexpr inline auto shell(const Unit<Shell> &u) {
  return u;
}

template<typename Shell>
constexpr int nbf(const Unit<Shell> &u) {
  return nbf(shell(u));
}

template<typename Shell>
LIBINTX_GPU_ENABLED
inline constexpr auto exp(const Unit<Shell>&, int k) {
  return std::integral_constant<int,0>{};
}

template<typename Shell>
LIBINTX_GPU_ENABLED
constexpr inline auto center(const Unit<Shell> &u) {
  return array<double,3>{ 0, 0, 0 };
}


}

#endif /* LIBINTX_SHELL_H */
