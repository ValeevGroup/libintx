#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "boys/reference.h"
#include "boys/asymptotic.h"
#include "boys/chebyshev.h"

using doctest::Approx;

TEST_CASE("asymptotic") {
  boys::Reference reference;
  for (double X : {117.0, 133.0, 200.0}) {
    for (size_t m = 0; m < 10; ++m) {
      CHECK(boys::asymptotic(X, m) == Approx(reference.compute(X,m)));
    }
    for (size_t k = 1; k < 10; ++k) {
      double s[10];
      boys::asymptotic(X, k, s);
      for (size_t m = 0; m < 10; ++m) {
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
    for (size_t m = 0; m < 10; ++m) {
      CHECK(chebyshev->compute(X,m) == Approx(reference.compute(X,m)));
    }
  }
}
