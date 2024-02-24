#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/engine/md/reference.h"
#include "libintx/pure.transform.h"

#include "libintx/cuda/api/api.h"
#include "libintx/cuda/md/engine.h"

#include <unsupported/Eigen/CXX11/Tensor>

using namespace libintx;

void md_eri4_subcase(int A, int B, int C, int D, std::pair<int,int> K = {1,1}) {

  namespace cuda = libintx::cuda;

  printf("(%i%i|%i%i) K={%i,%i}\n", A, B, C, D, K.first, K.second);

  int M = 16+3;
  int N = 16+1;

  int NA = npure(A);
  int NB = npure(B);
  int NC = npure(C);
  int ND = npure(D);

  auto [bra,ijs] = test::basis2({A,B}, {K.first,1}, M);
  auto [ket,kls] = test::basis2({C,D}, {K.second,1}, N);

  Eigen::Tensor<double,6> result(ND,NC,N,NB,NA,M);
  cuda::host::register_pointer(result.data(), result.size());

  cudaStream_t stream = 0;
  auto md = cuda::md::eri<4>(bra, ket, stream);
  md->compute(ijs, kls, result.data());
  cuda::stream::synchronize(stream);

  for (size_t ij = 0; ij < ijs.size(); ++ij) {
    for (size_t kl = 0; kl < kls.size(); ++kl) {

      auto [i,j] = ijs[ij];
      auto [k,l] = kls[kl];
      Eigen::Tensor<double,4> ab_cd_ref(npure(A), npure(B), npure(C), npure(D));
      {
        Eigen::Tensor<double,4> ab_cd_cartesian(
          ncart(A), ncart(B), ncart(C), ncart(D)
        );
        ab_cd_cartesian.setZero();
        libintx::md::reference::compute_ab_cd(bra[i], bra[j], ket[k], ket[l], ab_cd_cartesian);
        libintx::pure::reference::transform(
          A, B, C, D,
          ab_cd_cartesian,
          ab_cd_ref
        );
      }
      test::check4(
        [&](const auto &ab_cd_ref, auto i, auto j, auto k, auto l) {
          //printf("(%i,%i) %p\n", p, cd, &pCD(ij,p,cd,kl));
          auto ab_cd = result(l,k,(int)kl,j,i,(int)ij);
          CHECK(ab_cd == ab_cd_ref);
        },
        ab_cd_ref
      );
    }
  }

  cuda::host::unregister_pointer(result.data());

}

#define MD_ERI4_SUBCASE(A,B,C,D,Ks)                     \
  SUBCASE("(AB|CD)=(" # A # B "|" # C # D ")") {        \
    if (test::enabled(A,B,C,D)) {                       \
      for (auto K : Ks) {                               \
        md_eri4_subcase(A,B,C,D,K);                     \
      }                                                 \
    }                                                   \
  }

TEST_CASE("cuda.md.eri4") {
  std::vector< std::pair<int,int> > Ks = {
    {1,1},  {1,3}, {5,3}
  };
  MD_ERI4_SUBCASE(0,0,0,0,Ks);
  MD_ERI4_SUBCASE(1,0,0,0,Ks);
  MD_ERI4_SUBCASE(2,0,0,0,Ks);
  MD_ERI4_SUBCASE(0,0,2,0,Ks);
  MD_ERI4_SUBCASE(1,1,0,0,Ks);
  MD_ERI4_SUBCASE(1,1,2,0,Ks);
  MD_ERI4_SUBCASE(2,1,2,2,Ks);
}