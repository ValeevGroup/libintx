#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/engine/os/engine.h"
#include "libintx/reference.h"
#include "libintx/utility.h"

using namespace libintx;

constexpr double tolerance = 1e-6;

const Double<3> r0 = {  0.7, -1.2, -0.1 };
const Double<3> r1 = { -1.0,  0.0,  0.3 };
const Double<3> rx = {  0.5, -1.5,  0.9 };

auto obara_saika_eri3_test_case(
  int A, int B, int X,
  std::array<int,2> K = {1,1}, int N = 1)
{

  using libintx::test::gaussian;

  auto a = gaussian(A, K[0], false);
  auto b = gaussian(B, K[1], true);
  auto x = gaussian(X, 1, true);

  auto kernel = libintx::os::eri(a,b,x);

  int AB = nbf(a)*nbf(b);

  for (int i = 0; i < N; ++i) {
    kernel->compute(r0, r1, rx);
  }
  auto result = kernel->buffer();

  auto ref = libintx::reference::eri(a,b,x);
  const auto *reference = ref->compute(r0, r1, rx);

  for (int i = 0; i < AB; ++i) {
    for (int k = 0; k < nbf(x); ++k) {
      auto ab_x = result[k+i*nbf(x)];
      test::ReferenceValue ab_x_reference(
        reference[k+i*nbf(x)],
        tolerance,
        i%nbf(a), i/nbf(b), k
      );
      CHECK(ab_x == ab_x_reference);
    }
  }

}

#define ERI3_TEST_CASE(F,I,J,K)                 \
  if (test::enabled(I,J,K)) {                   \
    SUBCASE(str("(ab|x)=",I,J,K).c_str()) {     \
      for (int ka = KMAX; ka; ka /= 2) {        \
        for (int kb = KMAX; kb; kb /= 2) {      \
          F(I, J, K, {ka,kb});                  \
        }                                       \
      }                                         \
    }                                           \
  }

TEST_CASE("eri3") {
  for (size_t x = 0; x <= XMAX; ++x) {
    for (size_t a = 0; a <= LMAX; ++a) {
      for (size_t b = 0; b <= a; ++b) {
        ERI3_TEST_CASE(obara_saika_eri3_test_case, a,b,x);
      }
    }
  }
}
