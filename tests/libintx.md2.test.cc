#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/ao/md/reference.h"
#include "libintx/ao/md/engine.h"
#include "libintx/pure.reference.h"

using namespace libintx;
using test::zeros;

template<typename Operator>
auto params(Operator op) {
  if constexpr (op == Nuclear) {
    std::vector< std::tuple<int, std::array<double,3> > > params;
    for (size_t i = 0; i < 8+1; ++i) {
      params.push_back(
        {
          test::random<int>(1,100),
          test::random<double,3>(-1,+1)
        }
      );
    }
    return params;
  }
  else {
    return None;
  }
}

template<typename Operator>
void md2_test_subcase(Operator op, int A, int B, std::pair<int,int> K = {1,1}) {

  printf("(%i|%i) K={%i,%i}\n", A, B, K.first, K.second);

  int M = 32+3;
  int N = 16+1;

  int NA = npure(A);
  int NB = npure(B);

  auto [basis,ijs] = test::make_basis<2>({A,B}, {K.first,K.second}, M*N);

  auto result = zeros(ijs.size(),NA,NB);
  auto md = libintx::ao::integral_engine<2>(basis, basis);
  auto op_params = params(op);
  if constexpr (op == Nuclear) {
    md->set(typename Operator::Operator::Parameters{op_params});
  }
  md->compute(op,ijs,result.data());

  for (size_t ij = 0; ij < ijs.size(); ++ij) {
    auto [i,j] = ijs[ij];
    auto ab_ref = zeros(npure(A), npure(B));
    {
      auto ab_cartesian = zeros(ncart(A), ncart(B));
      libintx::md::reference::compute2<op>(basis[i], basis[j], op_params, ab_cartesian);
      libintx::pure::reference::transform(
        A, B,
        ab_cartesian,
        ab_ref
      );
    }
    test::check2(
      [&](auto ref, auto ... idx) {
        //printf("(%i) = %f\n", ij, result(ij,idx...));
        CHECK(result(ij,idx...) == ref.epsilon(1e-10));
      },
      ab_ref
    );
  }

}

#define LIBINTX_MD2_TEST_SUBCASE(Operator,A,B,Ks)               \
  if (test::enabled(A,B)) {                                     \
    SUBCASE(str(#Operator " (A|B)=(",A,B,")").c_str()) {        \
      printf(# Operator "\n");                                  \
      for (auto K : Ks) {                                       \
        md2_test_subcase(Operator,A,B,K);                       \
      }                                                         \
    }                                                           \
  }

std::vector< std::pair<int,int> > Ks = {
  {1,1}, {1,5}, {3,5}
};

#define LIBINTX_MD2_TEST_CASE(Operator)                 \
  TEST_CASE("libintx.md2." # Operator) {                \
    for (int a = 0; a <= LMAX; ++a) {                   \
      for (int b = 0; b <= LMAX; ++b) {                 \
        LIBINTX_MD2_TEST_SUBCASE(Operator,a,b,Ks);      \
      }                                                 \
    }                                                   \
  }

LIBINTX_MD2_TEST_CASE(Overlap);
LIBINTX_MD2_TEST_CASE(Kinetic);
LIBINTX_MD2_TEST_CASE(Nuclear);
