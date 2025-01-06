#ifndef LIBINTX_SHELL_H
#define LIBINTX_SHELL_H

#include "libintx/orbital.h"
#include "libintx/array.h"
#include "libintx/math.h"
#include "libintx/config.h"

#include <cassert>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace libintx::gto {

  template<typename T>
  struct Primitive;

  template<typename Shell, typename T = double>
  struct alignas(32) Gaussian;

} // libintx::gto

namespace libintx {

using Gaussian = gto::Gaussian<Shell>;

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

LIBINTX_GPU_ENABLED
constexpr inline int nbf(const Shell &a, const Shell &b) {
  return nbf(a)*nbf(b);
}

template<typename Shell>
struct Basis {

  Basis() = default;

  // explicit Basis(const std::vector<Shell> &shells) {
  //   this->assign(shells.begin(), shells.end());
  // }

  void push_back(const Shell &s) {
    this->shells_.push_back(s);
    using libintx::nbf;
    int first = (int)this->nbf_;
    int last = first + nbf(s);
    ranges_.push_back({first,last});
    this->nbf_ = (size_t)last;
  }

  auto empty() const { return shells_.empty(); }
  auto size() const { return shells_.size(); }
  auto nbf() const { return nbf_; }

  auto begin() const { return shells_.begin(); }
  auto end() const { return shells_.end(); }
  const auto& operator[](size_t i) const { return shells_.at(i); }

  const auto& ranges() const { return ranges_; }
  auto range(size_t i) const { return ranges_.at(i); }

  void erase(auto &&f) {
    auto shells = std::move(shells_);
    *this = Basis{};
    erase_if(shells, f);
    for (auto &s : shells) {
      this->push_back(s);
    }
  }

  auto max() const {
    struct Max {
      int L = -1;
      int K = 0;
    } max;
    for (auto &s : this->shells_) {
      max.L = std::max(max.L, s.L);
      max.K = std::max(max.K, nprim(s));
    }
    return max;
  }

private:
  std::vector<Shell> shells_;
  size_t nbf_ = 0;
  std::vector< libintx::range<int> > ranges_;

};

template<typename Shell = Gaussian, typename Z, typename R, typename ... Args>
auto make_basis(
  const std::vector< std::tuple<Z,R> > &atoms,
  const auto &basis_set,
  bool normalize = true)
{
  Basis<Shell> basis;
  for (auto [z,r] : atoms) {
    for (auto& [L,p] : basis_set.at(z)) {
      auto& [ r0,r1,r2 ] = r;
      Shell g(L, {r0,r1,r2}, p);
      if (normalize) g = normalized<Shell>(g);
      basis.push_back(g);
    }
  }
  return basis;
}

template<typename T, typename Shell>
std::vector<T> make_basis(
  const Basis<Shell> &first,
  const Basis<Shell> &second,
  const std::vector<Index2> &idx)
{
  std::vector<T> v;
  v.reserve(idx.size());
  for (auto &[i,j] : idx) {
    v.push_back(T{first[i], second[j] });
  }
  return v;
}

}

namespace libintx::gto {

template<typename T>
struct Primitive {
  T a = 1, C = NAN;
};

template<typename T, int KMAX>
auto make_primitives(const std::vector< Primitive<T> > &v) {
  if (v.size() > KMAX) {
    throw std::domain_error("libintx::gto::make_primitives: K > KMAX");
  }
  libintx::static_vector<Primitive<T>,KMAX> primitives = {{ }, v.size() };
  std::copy(v.begin(), v.end(), primitives.data);
  return primitives;
}


template<typename Shell, typename T>
struct alignas(32) Gaussian : Shell {

  constexpr static int KMAX = libintx::KMAX;

  using Primitive = gto::Primitive<T>;

  array<T,3> r;
  static_vector<Primitive,KMAX> prims = {};
  int K = 0;

  //LIBINTX_GPU_ENABLED
  Gaussian() = default;

  Gaussian(int L, const array<T,3> &r, std::vector<Primitive> ps, bool pure = true)
    : Shell({L,(int)pure}), r(r), prims(make_primitives<T,KMAX>(ps)), K(prims.size())
  {
  }

};

template<typename T>
auto normalized(int L, Primitive<T> p) {
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
  return Primitive<T>{alpha, C };
}

template<typename S, typename T>
Gaussian<Shell,T> normalized(const Gaussian<Shell,T> &s) {
  std::vector<Primitive<T>> prims(s.K);
  for (int i = 0; i < s.K; ++i) {
    prims[i] = normalized(s.L, s.prims[i]);
  }
  return Gaussian<Shell,T>(s.L, s.r, prims, s.pure);
}

template<typename S, typename T>
auto normalization_factor(const Gaussian<Shell,T> &g) {
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

template<typename Shell>
constexpr auto orbitals(const Shell &g) {
  return cartesian::orbitals(g.L);
}

template<typename Shell>
LIBINTX_GPU_ENABLED
inline constexpr auto nprim(const Shell &g) {
  return g.K;
}

template<typename Shell>
LIBINTX_GPU_ENABLED
inline constexpr auto primitives(const Shell &g) {
  return g.prims;
}

template<typename Shell>
LIBINTX_GPU_ENABLED
inline constexpr auto& primitive(const Shell &g, int k) {
  return g.prims[k];
}

template<typename Shell>
LIBINTX_GPU_ENABLED
inline constexpr auto& exp(const Shell &g, int k) {
  return g.prims[k].a;
}

template<typename Shell>
LIBINTX_GPU_ENABLED
inline constexpr auto& coeff(const Shell &g, int k) {
  return g.prims[k].C;
}

template<typename Shell>
LIBINTX_GPU_ENABLED
constexpr auto& center(const Shell &g) {
  return g.r;
}

template<typename T>
// G = std::function<const Gaussian*(size_t)>;
array<T,3> pack_centers(auto &&G) {
  using math::infinity;
  if constexpr (std::is_scalar_v<T>) {
    auto *g = G(0);
    if (g) return g->r;
    return { infinity<T>, infinity<T>, infinity<T> };
  }
  else {
    array<T,3> v = {};
    for (size_t i = 0; i < T::size(); ++i) {
      auto *g = G(i);
      if (!g) {
        for (int j = 0; j < 3; ++j) {
          constexpr auto inf = infinity<typename T::value_type>;
          v[j][i] = inf;
        }
        continue;
      }
      for (int j = 0; j < 3; ++j) {
        v[j][i] = g->r[j];
      }
    }
    return v;
  }
}

template<typename T>
// G = std::function<const Gaussian*(size_t)>;
auto pack_primitives(auto &&G) {
  auto *g = G(0);
  using primitives_t = decltype(g->prims);
  if constexpr (std::is_scalar_v<T>) {
    if (!g) return primitives_t{};
    return g->prims;
  }
  else {
    static_vector< gto::Primitive<T>, primitives_t::capacity()> v = { {}, g->prims.size() };
    for (size_t i = 0; i < T::size(); ++i) {
      auto *g = G(i);
      if (!g) continue;
      for (size_t k = 0; k < g->prims.size(); ++k) {
        auto [a,C] = primitives(*g)[k];
        v[k].a[i] = a;
        v[k].C[i] = C;
      }
    }
    return v;
  }
}


} // libintx::gto

namespace libintx {

template<typename S>
struct Unit : libintx::Shell {
  using Primitive = typename S::Primitive;
  static constexpr array<Primitive,1> prims = {{{ 0, 1 }}};
  static constexpr int K = 1;
  constexpr Unit() : libintx::Shell({0,true}) {}
};

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
