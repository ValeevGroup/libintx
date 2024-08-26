#include "libintx/engine/rysq/engine.h"
#include "libintx/pure.h"

#include "rysq/rysq.h"
//#define RYSQ_VECTORIZE
#include "rysq/simd.h"
#include "rysq/kernel.tcc"

namespace libintx {
namespace rysq {

  auto cast(const Double<3> &r) {
    return ::rysq::Vector<double,3>{ r[0], r[1], r[2] };
  }

  auto cast(const Gaussian &s) {
    std::vector<::rysq::Shell::Primitive> prims;
    for (int k = 0; k < s.K; ++k) {
      const auto &p = s.prims[k];
      prims.push_back({p.a, p.C});
    }
    return ::rysq::Shell{s.L, prims};
  }

  typedef std::function<void(const double*,double*)> Transform;

  inline Transform transform(const Shell &a, const Shell &b, const Shell &x) {
    return [a,b,x](const double *T, double *S) {
      for (int i = 0; i < ncart(a); ++i) {
        double *Si = S + i*nbf(b)*nbf(x);
        for (int j = 0; j < ncart(b); ++j) {
          double Sk[ncart(RYSQ_MAX_X)];
          for (int k = 0; k < ncart(x); ++k) {
            double t = T[i+j*ncart(a)+k*ncart(a)*ncart(b)];
            Sk[k] = t;
          }
          if (x.pure) cartesian_to_pure(x.L, Sk);
          double *Sij = Si + nbf(x)*(j);
          for (int k = 0; k < nbf(x); ++k) {
            Sij[k] = Sk[k];
          }
        }
        if (!b.pure) continue;
        for (int k = 0; k < nbf(x); ++k) {
          cartesian_to_pure(b.L, Si+k, nbf(x));
        }
      }
      if (!a.pure) return;
      for (int jk = 0; jk < nbf(b)*nbf(x); ++jk) {
        cartesian_to_pure(a.L, S+jk, nbf(b)*nbf(x));
      }
    };
  }

  struct Kernel3 : Kernel<3> {
    explicit Kernel3(const Gaussian& a, const Gaussian& b, const Gaussian &x) {
      this->impl_ = ::rysq::kernel( { cast(a), cast(b) }, { cast(x) });
      this->buffer_.reset(new double[ncart(a)*ncart(b)*nbf(x)]);
      transform_ = transform(a,b,x);
    }
    const double* compute(const Double<3> &a, const Double<3> &b, const Double<3> &x) override {
      assert(impl_);
      const double *eri = impl_->compute({ cast(a), cast(b), cast(x) });
      transform_(eri, buffer_.get());
      return this->buffer_.get();
    }
    void repeat(size_t n, const Double<3> &a, const Double<3> &b, const Double<3> &x) override {
      assert(impl_);
      auto a_ = cast(a);
      auto b_ = cast(b);
      auto x_ = cast(x);
      for (size_t i = 0; i < n; ++i) {
        impl_->compute(a_, b_, x_);
      }
    }
    const double* buffer() override {
      return this->buffer_.get();
    }
  private:
    std::unique_ptr<double[]> buffer_;
    std::unique_ptr<::rysq::Kernel3> impl_;
    Transform transform_;
  };

  struct Kernel4 : Kernel<4> {
    explicit Kernel4(std::unique_ptr<::rysq::Kernel4> impl)
      : impl_(std::move(impl)) {}
    const double* compute(
      const Double<3> &a, const Double<3> &b,
      const Double<3> &c, const Double<3> &d) override
    {
      assert(impl_);
      buffer_ = impl_->compute({ cast(a), cast(b), cast(c), cast(d) });
      return this->buffer();
    }
    void repeat(
      size_t n,
      const Double<3> &a, const Double<3> &b,
      const Double<3> &c, const Double<3> &d) override
    {
      assert(impl_);
      auto a_ = cast(a);
      auto b_ = cast(b);
      auto c_ = cast(c);
      auto d_ = cast(d);
      for (size_t i = 0; i < n; ++i) {
        impl_->compute(a_, b_, c_, d_);
      }
    }
    const double* buffer() override {
      return this->buffer_;
    }
  private:
    std::unique_ptr<::rysq::Kernel4> impl_;
    const double *buffer_;
  };

  std::unique_ptr< Kernel<3> > eri(const Gaussian& a, const Gaussian& b, const Gaussian& x) {
    return std::make_unique<Kernel3>(a,b,x);
  }

  std::unique_ptr< Kernel<4> > eri(
    const Gaussian& a, const Gaussian& b,
    const Gaussian& c, const Gaussian& d)
  {
    return std::make_unique<Kernel4>(
      ::rysq::kernel(
        { cast(a), cast(b) },
        { cast(c), cast(d) }
      )
    );
  }

}
}
