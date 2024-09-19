#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/ao/md/engine.h"
#include "libintx/ao/md/reference.h"
#include "libintx/pure.reference.h"

using namespace libintx;
using libintx::test::zeros;

int sample = 13;

template<typename Operator>
void libintx_md3_test_subcase(Operator op, int X, int C, int D, BraKet<int> K) {

  int M = 11;
  int N = 9;

  printf("* (%i|%i%i) dims=[%i,%i]\n", X, C, D, M, N);

  int NX = npure(X);
  int NC = npure(C);
  int ND = npure(D);

  auto [bra,xs] = test::make_basis<1>({X}, {K.bra}, M);
  auto [ket,kls] = test::make_basis<2>({C,D}, {K.ket,1}, N);

  std::array<size_t,2> dims = { (size_t)M*NX, (size_t)NC*ND*N };
  auto result = zeros(M, NX, NC, ND, N);
  libintx::md::IntegralEngine<3> md(
    std::make_shared< Basis<Gaussian> >(bra),
    std::make_shared< Basis<Gaussian> >(ket)
  );
  md.compute(op, xs, kls, {}, result.data(), dims);

  for (size_t kl = 0, ikl = 0; kl < kls.size(); ++kl) {
    for (size_t i = 0; i < xs.size(); ++i, ++ikl) {

      if (i%sample) continue;

      auto [k,l] = kls[kl];

      auto a = bra[i];
      auto b = Unit<Gaussian>{};
      auto c = ket[k];
      auto d = ket[l];
      auto ab_cd_ref = zeros(npure(X), 1, npure(C), npure(D));
      {
        auto ab_cd_cartesian = zeros(ncart(X), 1, ncart(C), ncart(D));
        libintx::md::reference::compute(a, b, c, d, ab_cd_cartesian);
        libintx::pure::reference::transform(
          X, 0, C, D,
          ab_cd_cartesian,
          ab_cd_ref
        );
      }
      test::check3(
        [&](auto ref, auto ... idx) {
          //printf("%.10f\n", result(i,idx...,(int)kl));
          CHECK(result(i,idx...,(int)kl) == ref.epsilon(1e-9));
        },
        test::Tensor<double,3>(ab_cd_ref.reshape(std::array<std::ptrdiff_t,3>{NX, NC, ND}))
      );
    }
  }

}


std::vector< BraKet<int> > Ks = {
  {1,1}, {1,5}, {3,5}
};

void libintx_md3_test_case(Operator op) {
  for (auto K : Ks) {
    printf("K = [%i,%i]\n", K.bra,K.ket);
    for (int c = 0; c <= LMAX; ++c) {
      for (int d = 0; d <= c; ++d) {
        for (int x = 0; x <= XMAX; ++x) {
          if (!test::enabled(x,c,d)) continue;
          libintx_md3_test_subcase(op,x,c,d,K);
        }
      }
    }
    printf("---------\n");
  }
}

#define LIBINTX_MD3_TEST_CASE(Operator)         \
  TEST_CASE("libintx.md3." # Operator) {        \
    printf("Operator=" # Operator "\n");        \
    printf("---------\n");                      \
    libintx_md3_test_case(Operator);            \
  }

LIBINTX_MD3_TEST_CASE(Coulomb);
