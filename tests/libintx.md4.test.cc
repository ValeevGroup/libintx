#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/ao/md/engine.h"
#include "libintx/ao/md/reference.h"
#include "libintx/pure.reference.h"

using namespace libintx;
using test::zeros;

int sample = 13;

template<typename Operator>
void libintx_md4_test_subcase(Operator op, int A, int B, int C, int D, BraKet<int> K) {

  int NA = npure(A);
  int NB = npure(B);
  int NC = npure(C);
  int ND = npure(D);

  int M = 11;
  int N = 9;

  printf("* (%i%i|%i%i) dims=[%i,%i]\n", A, B, C, D, M, N);

  auto [bra,ijs] = test::make_basis<2>({A,B}, {K.bra,1}, M);
  auto [ket,kls] = test::make_basis<2>({C,D}, {K.ket,1}, N);

  std::array<size_t,2> dims = { (size_t)M*NA*NB, (size_t)NC*ND*N };
  auto result = zeros(M, NA, NB, NC, ND, N);
  libintx::md::IntegralEngine<4> md(
    std::make_shared< Basis<Gaussian> >(bra),
    std::make_shared< Basis<Gaussian> >(ket)
  );
  md.compute(op, ijs, kls, {}, result.data(), dims);

  for (int kl = 0, i = 0; kl < (int)kls.size(); ++kl) {
    for (int ij = 0; ij < (int)ijs.size(); ++ij, ++i) {

      if (i%sample) continue;

      auto [i,j] = ijs[ij];
      auto [k,l] = kls[kl];

      auto a = bra[i];
      auto b = bra[j];
      auto c = ket[k];
      auto d = ket[l];
      auto ab_cd_ref = zeros(npure(A), npure(B), npure(C), npure(D));
      {
        auto ab_cd_cartesian = zeros(ncart(A), ncart(B), ncart(C), ncart(D));
        ab_cd_cartesian.setZero();
        libintx::md::reference::compute(a, b, c, d, ab_cd_cartesian);
        libintx::pure::reference::transform(
          A, B, C, D,
          ab_cd_cartesian,
          ab_cd_ref
        );
      }
      test::check4(
        [&](auto ref, auto ... idx) {
          ref = ref.at(ij,idx...,kl);
          CHECK(result(ij,idx...,kl) == ref.epsilon(1e-9));
        },
        ab_cd_ref
      );
    }
  }

}

std::vector< BraKet<int> > Ks = {
  {1,1}, {1,5}, {3,5}
};

void libintx_md4_test_case(Operator op) {
  for (auto K : Ks) {
    printf("K = [%i,%i]\n", K.bra,K.ket);
    for (int a = 0; a <= LMAX; ++a) {
      for (int b = 0; b <= a; ++b) {
        for (int c = 0; c <= LMAX; ++c) {
          for (int d = 0; d <= c; ++d) {
            if (!test::enabled(a,b,c,d)) continue;
            libintx_md4_test_subcase(op,a,b,c,d,K);
          }
        }
      }
    }
    printf("---------\n");
  }
}

#define LIBINTX_MD4_TEST_CASE(Operator)         \
  TEST_CASE("libintx.md4." # Operator) {        \
    printf("Operator=" # Operator "\n");        \
    printf("---------\n");                      \
    libintx_md4_test_case(Operator);            \
  }

LIBINTX_MD4_TEST_CASE(Coulomb);
