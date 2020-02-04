#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/boys/reference.h"
#include "libintx/interpolate/chebyshev.h"

TEST_CASE("chebyshev interpolation") {
  using namespace boys;

  auto reference = boys::reference();

  typedef double Real;
  ChebyshevInterpolation<Real> chebyshev(15);

  int M = 40;
  Real delta = 1.0;
  int K = int(delta*30);
  Real eps = 4*std::numeric_limits<Real>::epsilon();

  for (int m = 0; m < M; ++m) {

    DOCTEST_INFO("Validating F(" << m << ")");

    auto f = [m,&reference](Real x) {
      return (Real)reference->compute(x,m);
    };

    for (Real a = 0; a < 117; a += delta) {
      Real b = a+delta;
      DOCTEST_INFO(
        "Validating F(" << m << ", t=["
        << (double)a << ":"
        << (double)b <<"])"
      );
      auto p = chebyshev.generate(f, a, b);
      for (int k = 0; k < K; ++k) {
        Real x = a+k*(b-a)/K;
        Real px = chebyshev.polyval(p, x, a, b);
        libintx::test::ReferenceValue fx(f(x), eps, m, a, k);
        CHECK(px == fx);
      }
    }

  }

}
