#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/ao/md/hermite.h"
#include "libintx/ao/md/r1.h"
#include "libintx/ao/md/reference.h"
#include "libintx/boys/chebyshev.h"

using namespace libintx;
using namespace libintx::md;
using doctest::Approx;

TEST_CASE("hermite") {

  double a = 3.14;
  double b = 2.74;
  double R[3] = { 1.29, -2.8, 3.4 };

  E2<3> E(3,3,a,b,R);

  // CHECK(reference::E(1,1,0,a,b,R[0]) == Approx(-0.028828110715203102));
  // CHECK(reference::E(1,1,1,a,b,R[0]) == Approx(0.0006537302131611276));
  // CHECK(reference::E(1,1,2,a,b,R[0]) == Approx(0.0006334595088770618));

  // CHECK(reference::E(0,1,0,a,b,R[0]) == Approx(0.060349758358182674));
  // CHECK(reference::E(0,1,1,a,b,R[0]) == Approx(0.007449483824394248));
  // CHECK(reference::E(0,1,2,a,b,R[0]) == Approx(0.0));

  // CHECK(reference::E(2,0,0,a,b,R[0]) == Approx(0.039105728741112955));
  // CHECK(reference::E(2,0,1,a,b,R[0]) == Approx(-0.008956103920307448));
  // CHECK(reference::E(2,0,2,a,b,R[0]) == Approx(0.0006334595088770618));

  // CHECK(reference::E(3,3,0,a,b,R[0]) == Approx(-0.0039687691));
  // CHECK(reference::E(3,3,1,a,b,R[0]) == Approx(0.0001359494));
  // CHECK(reference::E(3,3,2,a,b,R[0]) == Approx(0.0001318959));

  CHECK(reference::E(0,0,0,a,b,R[0]) == Approx(E(0,0,0,0)));
  CHECK(reference::E(1,0,0,a,b,R[0]) == Approx(E(1,0,0,0)));
  CHECK(reference::E(1,0,1,a,b,R[0]) == Approx(E(1,0,1,0)));
  CHECK(reference::E(2,0,0,a,b,R[0]) == Approx(E(2,0,0,0)));
  CHECK(reference::E(2,0,1,a,b,R[0]) == Approx(E(2,0,1,0)));
  CHECK(reference::E(2,0,2,a,b,R[0]) == Approx(E(2,0,2,0)));
  CHECK(reference::E(3,0,0,a,b,R[0]) == Approx(E(3,0,0,0)));
  CHECK(reference::E(3,0,1,a,b,R[0]) == Approx(E(3,0,1,0)));
  CHECK(reference::E(3,0,2,a,b,R[0]) == Approx(E(3,0,2,0)));
  CHECK(reference::E(3,0,3,a,b,R[0]) == Approx(E(3,0,3,0)));

  CHECK(reference::E(0,1,0,a,b,R[0]) == Approx(E(0,1,0,0)));
  CHECK(reference::E(0,1,1,a,b,R[0]) == Approx(E(0,1,1,0)));
  CHECK(reference::E(0,2,0,a,b,R[0]) == Approx(E(0,2,0,0)));
  CHECK(reference::E(0,2,1,a,b,R[0]) == Approx(E(0,2,1,0)));
  CHECK(reference::E(0,2,2,a,b,R[0]) == Approx(E(0,2,2,0)));

  for (int i = 0; i <= 3; ++i) {
    for (int j = 0; j <= 3; ++j) {
      for (int k = 0; k <= 2*3; ++k) {
        CAPTURE(i);
        CAPTURE(j);
        CAPTURE(k);
        CHECK(reference::E(i,j,k,a,b,R[0]) == Approx(E(i,j,k,0)));
      }
    }
  }

}

TEST_CASE("r1") {

  static boys::Chebyshev<7,40,117,117*7> boys;

  constexpr int L = 3;

  array<double,3> PQ = { 3, 1, 0 };

  double alpha = 3.56;
  double s[L+1] = {};

  boys.template compute<L+1>(alpha*norm(PQ), 0, s);

  for (size_t m = 0; m <= L; ++m) {
    s[m] *= pow(-2*alpha,m);
  }

  auto visitor = [&s,&PQ](auto r) {
    auto [x,y,z] = r.orbital.lmn;
    auto u = reference::R(x, y, z, 0, s, PQ.data);
    CHECK(r.value == u);
  };

  r1::visit<L,r1::DepthFirst>(visitor, PQ, s);

}
