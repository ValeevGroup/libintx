#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/engine/md/reference.h"
#include "libintx/pure.transform.h"

#include "libintx/gpu/api/api.h"
#include "libintx/gpu/md/engine.h"

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

  Eigen::Tensor<double,6> result(M,NA,NB,NC,ND,N);
  cuda::host::register_pointer(result.data(), result.size());

  cudaStream_t stream = 0;
  auto md = cuda::md::eri<4>(bra, ket, stream);
  md->compute(ijs, kls, result.data(), {M*NA*NB, NC*ND*N});
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
        libintx::md::reference::compute(bra[i], bra[j], ket[k], ket[l], ab_cd_cartesian);
        libintx::pure::reference::transform(
          A, B, C, D,
          ab_cd_cartesian,
          ab_cd_ref
        );
      }
      test::check4(
        [&](const auto &ab_cd_ref, auto ... idx) {
          //printf("(%i,%i) %p\n", p, cd, &pCD(ij,p,cd,kl));
          auto ab_cd = result(ij,idx...,kl);
          CHECK(ab_cd == ab_cd_ref);
        },
        ab_cd_ref
      );
    }
  }

  cuda::host::unregister_pointer(result.data());

}

#define MD_ERI4_SUBCASE(A,B,C,D,Ks)                     \
  if (test::enabled(A,B,C,D)) {                         \
    SUBCASE(str("(AB|CD)=(",A,B,"|",C,D,")").c_str()) { \
      for (auto K : Ks) {                               \
        md_eri4_subcase(A,B,C,D,K);                     \
      }                                                 \
    }                                                   \
  }

TEST_CASE("cuda.md.eri4") {

  std::vector< std::pair<int,int> > Ks = {
    {1,1}, {1,5}, {3,5}
  };

  // for (int ic = 0; ic <= LMAX; ++ic) {
  //   for (int id = 0; id <= ic; ++id) {
  //     for (int ia = 0; ia <= LMAX; ++ia) {
  //       for (int ib = 0; ib <= ia; ++ib) {
  //         MD_ERI4_SUBCASE(ia,ib,ic,id,Ks);
  //       }
  //     }
  //   }
  // }

  MD_ERI4_SUBCASE(1,0,0,0,Ks);
  MD_ERI4_SUBCASE(0,0,0,0,Ks);
  MD_ERI4_SUBCASE(1,0,0,0,Ks);
  MD_ERI4_SUBCASE(1,2,0,0,Ks);
  MD_ERI4_SUBCASE(2,0,0,0,Ks);
  MD_ERI4_SUBCASE(0,0,2,0,Ks);
  MD_ERI4_SUBCASE(1,1,0,0,Ks);
  MD_ERI4_SUBCASE(1,1,1,0,Ks);

  MD_ERI4_SUBCASE(1,1,2,0,Ks);
  MD_ERI4_SUBCASE(2,2,1,0,Ks);
  MD_ERI4_SUBCASE(1,0,2,2,Ks);
  MD_ERI4_SUBCASE(1,1,3,0,Ks);
  MD_ERI4_SUBCASE(3,3,1,0,Ks);
  MD_ERI4_SUBCASE(1,0,3,3,Ks);

  MD_ERI4_SUBCASE(1,1,1,1,Ks);
  MD_ERI4_SUBCASE(2,2,2,2,Ks);
  MD_ERI4_SUBCASE(3,3,3,3,Ks);



}
