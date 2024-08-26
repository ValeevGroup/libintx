#include "reference.h"
#include "libintx/shell.h"

#include <libint2/cxxapi.h>
#include <libint2/engine.h>

#include <memory>
#include <string>

namespace libintx::reference {

  constexpr double precision = 1e-10;

  void initialize() {
    ::libint2::initialize();
  }

  auto cast(const libintx::array<double,3> &r) {
    return std::array<double,3>{ r[0], r[1], r[2] };
  }

  auto cast(const Gaussian &g) {
    ::libint2::svector<::libint2::Shell::real_t> exponents, coeff;
    for (int i = 0; i < g.K; ++i) {
      auto p = g.prims[i];
      exponents.push_back(p.a);
      coeff.push_back(p.C);
    }
    if (exponents.empty()) throw;
    if (coeff.empty()) throw;
    std::array<double,3> center = {};
    ::libint2::Shell::do_enforce_unit_normalization(false);
    auto s = ::libint2::Shell(
      exponents,
      {{g.L, (bool)g.pure, coeff}},
      center
    );
    for (int k = 0; k < g.K; ++k) {
      s.contr[0].coeff[k] = g.prims[k].C;
      assert(s.contr[0].coeff[k] == g.prims[k].C);
    }
    return s;
  }

  template<::libint2::Operator Operator, ::libint2::BraKet BraKet>
  void compute(size_t n, const auto& args, double *data, size_t nbf) {
    auto engine = ::libint2::Engine(Operator, 10, LIBINT_MAX_AM);
    engine.set(BraKet);
    engine.set_precision(precision);
    for (size_t i = 0; i < n; ++i) {
      std::apply(
        [&](auto&& ... args) {
          engine.compute2<Operator, BraKet, 0>(args...);
        },
        args
      );
      if (data) {
        std::copy(engine.results()[0], engine.results()[0]+nbf, data+i*nbf);
      }
    }
  };

}

void libintx::reference::compute(
  size_t n,
  const Gaussian& a, const Gaussian& c, const Gaussian& d,
  double *data)
{
  size_t m = nbf(a)*nbf(c)*nbf(d);
  auto p = cast(a);
  auto q = ::libint2::Shell::unit();
  auto r = cast(c);
  auto s = cast(d);
  auto bra = ::libint2::ShellPair(p, q, std::log(precision));
  auto ket = ::libint2::ShellPair(r, s, std::log(precision));
  auto args = std::tuple(p, q, r, s, &bra, &ket);
  compute<::libint2::Operator::coulomb,::libint2::BraKet::xs_xx>(n, args, data, m);
}

void libintx::reference::compute(
  size_t n,
  const Gaussian& a, const Gaussian& b,
  const Gaussian& c, const Gaussian& d,
  double *data)
{
  size_t m = nbf(a)*nbf(b)*nbf(c)*nbf(d);
  auto p = cast(a);
  auto q = cast(b);
  auto r = cast(c);
  auto s = cast(d);
  auto bra = ::libint2::ShellPair(p, q, std::log(precision));
  auto ket = ::libint2::ShellPair(r, s, std::log(precision));
  auto args = std::tuple(p, q, r, s, &bra, &ket);
  compute<::libint2::Operator::coulomb,::libint2::BraKet::xx_xx>(n, args, data, m);
}
