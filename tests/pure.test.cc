#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/pure.transform.h"
#include "libintx/pure.reference.h"

using libintx::test::ReferenceValue;
using namespace libintx;

TEST_CASE("Pure coefficients") {

  constexpr int L = 6;
  for (auto [x,y,z] : cartesian::orbitals<L>()) {
    for (auto [l,m] : pure::orbitals<L>()) {
      auto ref = ReferenceValue(pure::reference::coefficient(l,m,x,y,z)).at(l,m,x,y,z);
      CHECK(pure::coefficient(l,m,x,y,z) == ref.epsilon(1e-15));
    }
  }

}
