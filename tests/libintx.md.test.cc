#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/ao/md/hermite.h"
#include "libintx/ao/md/r1.h"
#include "libintx/ao/md/reference.h"
#include "libintx/boys/chebyshev.h"

using doctest::Approx;
using libintx::foreach;
using libintx::str;
using namespace libintx::test;

namespace reference = libintx::md::reference;

TEST_CASE("hermite.E2") {

  double a = 3.14;
  double b = 2.74;
  libintx::array<double,3> R = { 1.29, -2.8, 3.4 };

  auto test = [&](auto A, auto B, auto P) {
    libintx::md::E2<double,A,B,P> E(a,b,R);
    for (size_t i = 0; i <= A; ++i) {
      for (size_t j = 0; j <= B; ++j) {
        for (size_t k = 0; k <= P; ++k) {
          for (size_t ix = 0; ix < 3; ++ix) {
            auto ref = reference::E(i,j,k,a,b,R[ix]);
            CHECK(ReferenceValue(ref).at(A,B,P,i,j,k,ix) == E(i,j,k,ix));
          }
        }
      }
    }
  };

  constexpr int L = std::min(3,libintx::LMAX);
  libintx::foreach2(
    std::make_index_sequence<(L+1)*(L+1)>{},
    std::make_index_sequence<2*L+1>{},
    [&](auto AB, auto P) {
      auto A = std::integral_constant<size_t, AB%(L+1)>{};
      auto B = std::integral_constant<size_t, AB/(L+1)>{};
      test(A,B,P);
    }
  );

}

TEST_CASE("md.r1") {

  constexpr int L = std::min(3,libintx::LMAX);
  constexpr int M = 4*L;

  using namespace libintx;
  using namespace libintx::md;
  static boys::Chebyshev<7,M+1,117,117*7> boys;

  libintx::array<double,3> PQ = { 3, 1, 0 };
  double alpha = 3.56;

  libintx::foreach(
    std::make_index_sequence<M+1>{},
    [&](auto M) {
      double s[M+1] = {};
      boys.template compute<M>(alpha*norm(PQ), s);
      for (size_t m = 0; m <= M; ++m) {
        s[m] *= pow(-2*alpha,m);
      }
      auto visitor = [&s,&PQ,M](auto r) {
        auto [x,y,z] = r.orbital.lmn;
        auto u = reference::R(x, y, z, 0, s, PQ.data);
        CHECK(ReferenceValue(u).at(M, r.index) == r.value);
      };
      r1::visit<M>(visitor, PQ, s);
    }
  );

}
