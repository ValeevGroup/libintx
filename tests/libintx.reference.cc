#include "reference.h"
#include "test.h"
#include "libintx/shell.h"

#include <libint2/cxxapi.h>
#include <libint2/engine.h>

#include <memory>
#include <string>

namespace libintx::reference {

  constexpr double precision = 1e-100;

  void initialize() {
    ::libint2::initialize();
  }

  auto cast(const libintx::array<double,3> &r) {
    return std::array<double,3>{ r[0], r[1], r[2] };
  }

  auto make_point_charges(const std::any &params) {
    using Params = std::vector< std::tuple<int,std::array<double,3> > >;
    const auto &rs = std::any_cast<Params>(params);
    std::vector< std::pair<double,std::array<double,3> > > cs;
    for (auto [Z,r] : rs) {
      cs.emplace_back(Z,r);
    }
    return cs;
  }

  auto cast(const Gaussian &g) {
    ::libint2::svector<::libint2::Shell::real_t> exponents, coeff;
    for (int i = 0; i < g.K; ++i) {
      auto p = g.prims[i];
      exponents.push_back(p.a);
      coeff.push_back(p.C);
    }
    libintx_assert(!exponents.empty());
    libintx_assert(!coeff.empty());
    std::array<double,3> center = cast(g.r);
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

  template<::libint2::Operator Operator>
  double time1(auto &&params, size_t n, const auto& ab, double *data, size_t nbf) {
    auto engine = ::libint2::Engine(Operator, 10, LIBINT_MAX_AM);
    engine.set(libint2::BraKet::x_x);
    engine.set_precision(precision);
    engine.set_params(params);
    auto &[a,b] = ab;
    auto t = time::now();
    for (size_t i = 0; i < n; ++i) {
      engine.compute1(a,b);
      if (data) {
        std::copy(engine.results()[0], engine.results()[0]+nbf, data+i*nbf);
      }
    }
    return time::since(t);
  };

  template<::libint2::Operator Operator, ::libint2::BraKet BraKet>
  double time2(size_t n, auto &&A, auto &&B, auto &&C, auto &&D, auto *bra, auto *ket, double *data) {
    auto engine = ::libint2::Engine(Operator, 10, LIBINT_MAX_AM);
    engine.set(BraKet);
    engine.set_precision(precision);
    size_t nbf = A.size()*B.size()*C.size()*D.size();
    auto t = time::now();
    for (size_t i = 0; i < n; ++i) {
      engine.compute2<Operator, BraKet, 0>(A,B,C,D,bra,ket);
      if (data) {
        std::copy(engine.results()[0], engine.results()[0]+nbf, data+i*nbf);
      }
    }
    return time::since(t);
  };

}

double libintx::reference::time(
  Operator op, const std::any &params,
  size_t n, const Gaussian& a, const Gaussian& b,
  double *data)
{
  initialize();
  size_t m = nbf(a)*nbf(b);
  auto ab = std::tuple(cast(a), cast(b));
  if (op == Overlap) {
    return time1<::libint2::Operator::overlap>(nullptr, n, ab, data, m);
  }
  if (op == Kinetic) {
    return time1<::libint2::Operator::kinetic>(nullptr, n, ab, data, m);
  }
  if (op == Nuclear) {
    return time1<::libint2::Operator::nuclear>(make_point_charges(params), n, ab, data, m);
  }
  return 0;
}

double libintx::reference::time(
  Operator op, const std::any &params,
  size_t n,
  const Gaussian& a, const Gaussian& c, const Gaussian& d,
  double *data)
{
  libintx_assert(op == Coulomb);
  initialize();
  auto p = cast(a);
  auto q = ::libint2::Shell::unit();
  auto r = cast(c);
  auto s = cast(d);
  auto bra = ::libint2::ShellPair(p, q, std::log(precision));
  auto ket = ::libint2::ShellPair(r, s, std::log(precision));
  return time2<::libint2::Operator::coulomb,::libint2::BraKet::xs_xx>(n, p, q, r, s, &bra, &ket, data);
}

double libintx::reference::time(
  Operator op, const std::any &params,
  size_t n,
  const Gaussian& a, const Gaussian& b,
  const Gaussian& c, const Gaussian& d,
  double *data)
{
  //printf("K=%i,%i\n", a.K*b.K, c.K*d.K);
  libintx_assert(op == Coulomb);
  initialize();
  auto p = cast(a);
  auto q = cast(b);
  auto r = cast(c);
  auto s = cast(d);
  auto bra = ::libint2::ShellPair(p, q, std::log(precision));
  auto ket = ::libint2::ShellPair(r, s, std::log(precision));
  return time2<::libint2::Operator::coulomb,::libint2::BraKet::xx_xx>(n, p, q, r, s, &bra, &ket, data);
}
