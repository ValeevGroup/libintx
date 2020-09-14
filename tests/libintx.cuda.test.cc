#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/cuda/eri.h"
#include "libintx/engine/os/engine.h"
#include "libintx/reference.h"
#include "libintx/utility.h"

using namespace libintx;
using namespace libintx::cuda;

template<size_t N>
auto integral_list(size_t n) {
  return IntegralList<N>(n);
}

constexpr double tolerance = 1e-7;

const Double<3> r0 = {  0.7, -1.2, -0.1 };
const Double<3> r1 = { 0.75,  3.14,  -0.237 };
const Double<3> rx = {  0.5, -1.5,  0.9 };

//const Double<3> r0 = {  0, 0, 0 };
//const Double<3> r1 = {  0, 0, 0 };
//const Double<3> rx = {  0, 0, 0 };

auto obara_saika_eri3_test_case(
  int A, int B, int X,
  std::array<int,2> K = {1,1}, int N = 1)
{

  if (K[0]*K[1] > 1 && A+B+X >= 3*LMAX) return;

  auto a = test::gaussian(A, K[0], true);
  auto b = test::gaussian(B, K[1], true);
  auto x = test::gaussian(X, 1, true);

  auto engine = libintx::cuda::eri<3>();

  auto centers = std::vector< Double<3> >{r0,r1,rx};
  engine->set_centers(centers);

  int AB = nbf(a)*nbf(b);
  auto result = host::vector<double>(N*nbf(x)*AB);

  IntegralList<3> list(N);
  for (int i = 0; i < N; ++i) {
    double *output = result.data() + i*nbf(x)*AB;
    double scale = 1.0;
    list[i] = { {0,1,2}, output, scale };
  }

  engine->compute(a, b, x, list);
  device::synchronize();

  auto ref = libintx::reference::eri(a,b,x);
  ref->compute(r0, r1, rx);

  // auto ref = libintx::os::eri(a,b,x);
  // ref->compute(r0, r1, rx);

  for (int i = 0; i < AB; ++i) {
    for (int k = 0; k < nbf(x); ++k) {
      auto ab_x = result[k+i*nbf(x)];
      test::ReferenceValue ab_x_reference(
        ref->buffer()[k+i*nbf(x)],
        tolerance,
        i%nbf(a), i/nbf(b), k
      );
      CHECK(ab_x == ab_x_reference);
    }
  }

}

#define ERI3_TEST_CASE(F,I,J,K,...)                     \
  if (test::enabled(I,J,K)) {                           \
    SUBCASE(str("(ab|x)=",I,J,K).c_str()) {             \
      for (int ka = KMAX; ka; ka /= 2) {                \
        for (int kb = KMAX; kb; kb /= 2) {              \
          F(I, J, K, {ka,kb});                          \
        }                                               \
      }                                                 \
    }                                                   \
  }

TEST_CASE("obara-saika.eri3") {
  for (size_t x = 0; x <= XMAX; ++x) {
    for (size_t a = 0; a <= LMAX; ++a) {
      for (size_t b = 0; b <= a; ++b) {
        ERI3_TEST_CASE(obara_saika_eri3_test_case, a,b,x);
      }
    }
  }
}
