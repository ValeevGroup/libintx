#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/boys/reference.h"
#include "libintx/boys/asymptotic.h"
#include "libintx/boys/chebyshev.h"
#include "libintx/math/interpolate/chebyshev.h"

using doctest::Approx;
namespace boys = libintx::boys;

constexpr int MAXM = 12;

TEST_CASE("chebyshev interpolation") {

  auto reference = boys::reference();

  libintx::math::ChebyshevInterpolation<double> chebyshev(15);

  double delta = 1.0;
  int K = int(delta*30);
  double eps = 4*std::numeric_limits<double>::epsilon();

  for (int m = 0; m < MAXM; ++m) {

    DOCTEST_INFO("Validating F(" << m << ")");

    auto f = [m,&reference](double x) {
      return (double)reference->compute(x,m);
    };

    for (double a = 0; a < 117; a += delta) {
      double b = a+delta;
      DOCTEST_INFO(
        "Validating F(" << m << ", t=["
        << (double)a << ":"
        << (double)b <<"])"
      );
      auto p = chebyshev.generate(f, a, b);
      for (int k = 0; k < K; ++k) {
        double x = a+k*(b-a)/K;
        double px = chebyshev.polyval(p, x, a, b);
        CHECK(px == Approx(f(x)).epsilon(eps));
      }
    }

  }

}

TEST_CASE("asymptotic") {
  boys::Reference reference;
  for (double X : {117.0, 133.0, 200.0}) {
    for (size_t m = 0; m < MAXM; ++m) {
      CHECK(boys::asymptotic(X, m) == Approx(reference.compute(X,m)));
    }
    for (size_t k = 1; k < MAXM; ++k) {
      double s[10];
      boys::asymptotic(X, k, s);
      for (size_t m = 0; m < MAXM; ++m) {
        CHECK(s[m] == Approx(reference.compute(X,k+m)));
      }
    }
  }
}

TEST_CASE("chebyshev") {
  auto chebyshev = boys::chebyshev();
  boys::Reference reference;
  for (size_t i = 0; i < 300; ++i) {
    double X = (double)i/3;
    for (size_t m = 0; m < MAXM; ++m) {
      CHECK(chebyshev->compute(X,m) == Approx(reference.compute(X,m)));
    }
  }
}

#include "libintx/simd.h"

#ifdef LIBINTX_SIMD_DOUBLE

TEST_CASE("simd") {
  constexpr auto MAXM = 8;
  using simd = LIBINTX_SIMD_DOUBLE;
  auto &chebyshev = libintx::boys::chebyshev<MAXM+1>();
  libintx::boys::Reference reference;
  for (size_t i = 0; i < 10*36; i += simd::size()) {
    auto X = simd([&](auto Lane) { return (double)(i+Lane)/10; });
    simd Fm[MAXM+1] = {};
    chebyshev.compute<MAXM>(X,0,Fm);
    for (int m = 0; m <= MAXM; ++m) {
      for (size_t lane = 0; lane < simd::size(); ++lane) {
        auto ref = reference.compute(X[lane],m);
        CHECK((double)Fm[m][lane] == Approx(ref).epsilon(1e-15));
      }
      //break;
    }
  }
}

#endif
