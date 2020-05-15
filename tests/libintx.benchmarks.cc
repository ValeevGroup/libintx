#include "libintx/engine/os/engine.h"
#include "libintx/utility.h"
#include "test.h"

#ifdef LIBINTX_LIBINT2
#include "libintx/engine/libint2/engine.h"
#endif

using namespace libintx;

const Double<3> r0 = {  0.7, -1.2, -0.1 };
const Double<3> r1 = { -1.0,  0.0,  0.3 };
const Double<3> rx = {  0.5, -1.5,  0.9 };

auto kernel_test_case(int A, int B, int X, std::vector< std::array<int,2> > Ks, int N = 1) {

  printf("%i%i%i, ", A, B, X);

  using libintx::test::gaussian;

  auto x = gaussian(X, 1, true);

  for (auto K : Ks) {

    auto ak = gaussian(A, K[0], true);
    auto bk = gaussian(B, K[1], true);

    int Nk = N;///(A+B+X+1));

    double tx = 0;
    {
      auto kernel = libintx::os::eri(ak,bk,x);
      auto t0 = time::now();
      for (int i = 0; i < Nk; ++i) {
        kernel->compute(r0, r1, rx);
      }
      tx = time::since(t0);
      printf("K={%i,%i},t=%.4f,", K[0], K[1], tx);
    }

#ifdef LIBINTX_LIBINT2
    {
      auto libint2 = libintx::libint2::kernel(ak,bk,x);
      auto t0 = time::now();
      for (int i = 0; i < Nk; ++i) {
        libint2->compute(r0, r1, rx);
      }
      auto t = time::since(t0);
      printf("t(libint2)/t=%.3f,", t/tx);
    }
#endif

    printf(" ");

  }

  printf("\n");

}

#define KERNEL_TEST_CASE(I,J,K,...)                                       \
  if (test::enabled(I,J,K))  { kernel_test_case(I,J,K,__VA_ARGS__); }

void kernel_test_all(std::vector< std::array<int,2> > Ks, int N) {
  for (size_t x = 0; x <= XMAX; ++x) {
    for (size_t a = 0; a <= LMAX; ++a) {
      for (size_t b = 0; b <= a; ++b) {
        KERNEL_TEST_CASE(a,b,x, Ks, N);
      }
    }
  }
}

int main(int argc, const char **argv) {
  int N = (argc-1 ? std::stoi(argv[1]) : 1000);
  std::vector< std::array<int,2> > Ks = { {1,1}, {5,1}, {5,5} };
  kernel_test_all(Ks, N);
}
