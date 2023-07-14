#include "libintx/engine/libint2/engine.h"
#include <libint2/cxxapi.h>
#include <libint2/engine.h>

#include <memory>
#include <string>

namespace libintx {
namespace libint2 {

  auto cast(const Double<3> &r) {
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

  struct Kernel3 : Kernel<3> {
    explicit Kernel3(const Gaussian& a, const Gaussian& b, const Gaussian& x)
      : a_(cast(a)), b_(cast(b)), x_(cast(x))
    {
      ::libint2::initialize();
      auto impl = new ::libint2::Engine(::libint2::Operator::coulomb, 10, LIBINT_MAX_AM);
      impl->set(::libint2::BraKet::xx_xs);
      impl_.reset(impl);
    }
    const double* compute(const Double<3> &a, const Double<3> &b, const Double<3> &x) override {
      assert(impl_);
      impl_->compute(
        a_.move(cast(a)),
        b_.move(cast(b)),
        x_.move(cast(x))
        //::libint2::Shell::unit()
      );
      return this->buffer();
    }
    const double* buffer() override {
      return this->impl_->results()[0];
    }
  private:
    std::unique_ptr<::libint2::Engine> impl_;
    ::libint2::Shell a_, b_, x_;
  };

  struct Kernel4 : Kernel<4> {
    explicit Kernel4(const Gaussian& a, const Gaussian& b, const Gaussian& c, const Gaussian& d)
      : a_(cast(a)), b_(cast(b)), c_(cast(c)), d_(cast(d))
    {
      ::libint2::initialize();
      auto impl = new ::libint2::Engine(::libint2::Operator::coulomb, 10, LIBINT_MAX_AM);
      impl->set(::libint2::BraKet::xx_xx);
      impl_.reset(impl);
    }
    const double* compute(const Double<3> &a, const Double<3> &b, const Double<3> &c, const Double<3> &d) override {
      assert(impl_);
      impl_->compute(
        a_.move(cast(a)),
        b_.move(cast(b)),
        c_.move(cast(c)),
        d_.move(cast(d))
      );
      return this->buffer();
    }
    const double* buffer() override {
      return this->impl_->results()[0];
    }
  private:
    std::unique_ptr<::libint2::Engine> impl_;
      ::libint2::Shell a_, b_, c_, d_;
  };

  std::unique_ptr< Kernel<3> > kernel(const Gaussian& a, const Gaussian& b, const Gaussian& x) {
    return std::make_unique<Kernel3>(a,b,x);
  }

  std::unique_ptr< Kernel<4> > kernel(const Gaussian& a, const Gaussian& b, const Gaussian& c, const Gaussian& d) {
    return std::make_unique<Kernel4>(a,b,c,d);
  }

}
}
