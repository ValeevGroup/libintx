#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "boys/asymptotic.h"
#include "boys/reference.h"

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
