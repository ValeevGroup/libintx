#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/engine/md/reference.h"
#include "libintx/pure.transform.h"

#include "libintx/gpu/api/api.h"
#include "libintx/gpu/md/engine.h"

#include <unsupported/Eigen/CXX11/Tensor>

using namespace libintx;

void md_eri3_subcase(int X, int C, int D, std::pair<int,int> K = {1,1}) {

  namespace cuda = libintx::cuda;

  printf("(%i|%i%i) K={%i,%i}\n", X, C, D, K.first, K.second);

  int M = 16+3;
  int N = 16+1;

  int NX = npure(X);
  int NC = npure(C);
  int ND = npure(D);

  auto [bra,is] = test::basis1({X}, {K.first}, M);
  auto [ket,kls] = test::basis2({C,D}, {K.second,1}, N);

  Eigen::Tensor<double,6> result(M,NX,1,NC,ND,N);
  cuda::host::register_pointer(result.data(), result.size());

  cudaStream_t stream = 0;
  auto md = libintx::cuda::md::eri<3>(bra, ket, stream);
  md->compute(is, kls, result.data(), {(size_t)M*NX, (size_t)N*NC*ND});
  cuda::stream::synchronize(stream);

  for (size_t i = 0; i < bra.size(); ++i) {
    for (size_t kl = 0; kl < kls.size(); ++kl) {

      auto [k,l] = kls[kl];
      Eigen::Tensor<double,4> x_cd_ref(npure(X), 1, npure(C), npure(D));
      {
        Eigen::Tensor<double,4> x_cd_cartesian(
          ncart(X), 1, ncart(C), ncart(D)
        );
        x_cd_cartesian.setZero();
        libintx::md::reference::compute(
          bra[i], Unit<Gaussian>{},
          ket[k], ket[l],
          x_cd_cartesian
        );
        //std::cout << x_cd_cartesian << std::endl;
        libintx::pure::reference::transform(
          X, 0, C, D,
          x_cd_cartesian,
          x_cd_ref
        );
      }
      test::check4(
        [&](const auto &x_cd_ref, auto ... idx) {
          //printf("(%i,%i) %p\n", p, cd, &pCD(ij,p,cd,kl));
          auto x_cd = result(i,idx...,kl);
          CHECK(x_cd == x_cd_ref);
        },
        x_cd_ref
      );
    }
  }

  cuda::host::unregister_pointer(result.data());

}

#define MD_ERI3_SUBCASE(X,C,D,Ks)                       \
  if (test::enabled(X,C,D)) {                           \
    SUBCASE(str("(X|CD)=(",X,"|",C,D,")").c_str()) {    \
      for (auto K : Ks) {                               \
        md_eri3_subcase(X,C,D,K);                       \
      }                                                 \
    }                                                   \
  }

TEST_CASE("cuda.md.eri3") {

  std::vector< std::pair<int,int> > Ks = {
    {1,1}, {1,3} // , {3,5}
  };

  for (int id = 0; id <= LMAX; ++id) {
    for (int ic = 0; ic <= LMAX; ++ic) {
      for (int ix = 0; ix <= XMAX; ++ix) {
        MD_ERI3_SUBCASE(ix,ic,id,Ks);
      }
    }
  }

}
