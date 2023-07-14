#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/engine/md/reference.h"
#include "libintx/boys/reference.h"
#include "libintx/engine/md/r1/recurrence.h"
#include "libintx/reference.h"

#include "libintx/cuda/api/api.h"
#include "libintx/cuda/api/thread_group.h"
#include "libintx/cuda/md/engine.h"

using namespace libintx;
using doctest::Approx;

template<typename F, typename ... Args>
__global__
void md_test(F f, Args ... args) {
  f(args...);
}

// __device__
// static constexpr libintx::md::r1::Recurrence<24> recurrence;

template<int L, int Block>
void md_r1_test() {

  // using namespace libintx;
  // namespace cuda = libintx::cuda;

  // constexpr double tolerance = 1e-10;

  // Double<3> PQ = { 13.7, 1, -6.626 };
  // Double<L+1> s = {};

  // for (size_t m = 0; m <= L; ++m) {
  //   double alpha = 0.0314;
  //   double Fm = boys::Reference().compute(alpha*norm(PQ), m);
  //   //double Fm = test::random<double>();
  //   s[m] = Fm*pow(-2*alpha,m);
  // }

  // cuda::host::vector<double> r1(nherm2(L));
  // for (size_t m = 0; m <= L; ++m) {
  //   r1[m] = s[m];
  // }

  // md_test<<<1,Block>>>(
  //   [] __device__ (const Double<3> &PQ, double *R) {
  //     namespace r1 = libintx::md::r1;
  //     r1::compute<L>(recurrence, PQ, R, cuda::thread_block<Block>());
  //   },
  //   PQ,
  //   r1.data()
  // );
  // cuda::device::synchronize();

  // for (auto h : hermitian::orbitals<L>) {
  //   auto [x,y,z] = h;
  //   double r = r1[hermitian::index2(h)];
  //   test::ReferenceValue reference(
  //     md::reference::R(x, y, z, 0, s.data, PQ.data),
  //     tolerance,
  //     x,y,z
  //   );
  //   CHECK(r == reference);
  // }

}

#define MD_R1_SUBCASE(L,B) SUBCASE("L=" # L " B=" # B) { md_r1_test<L,B>(); }

// TEST_CASE("cuda.md.r1") {
//   MD_R1_SUBCASE(1,32);
//   MD_R1_SUBCASE(2,32);
//   MD_R1_SUBCASE(3,32);
//   MD_R1_SUBCASE(4,32);
//   MD_R1_SUBCASE(5,32);
//   MD_R1_SUBCASE(6,32);
//   MD_R1_SUBCASE(15,4*32);
//   MD_R1_SUBCASE(16,8*32);
//   MD_R1_SUBCASE(17,8*32);
//   MD_R1_SUBCASE(18,8*32);
//   MD_R1_SUBCASE(19,8*32);
//   MD_R1_SUBCASE(23,4*32);
//   MD_R1_SUBCASE(24,8*32);
// }

template<int A, int B, int C, int D>
void md_eri_test(std::pair<int,int> K = {1,1}) {

  namespace cuda = libintx::cuda;
  using libintx::Index2;

  bool pure = true;
  Basis<Gaussian> bra;
  Basis<Gaussian> ket;

  printf("(%i%i|%i%i) K={%i,%i}\n", A, B, C, D, K.first, K.second);

  int nab = 0, ncd = 0;
  std::vector<Index2> ij, kl;

  for (int i = 0; i < 5; ++i) {
    auto a = test::gaussian(A, K.first, pure);
    auto b = test::gaussian(B, 1, pure);
    auto r0 = test::random<double,3>(-0.25,0.25);
    auto r1 = test::random<double,3>(-0.25,0.25);
    bra.push_back({a,r0});
    bra.push_back({b,r1});
    ij.push_back(Index2{i*2,i*2+1});
    nab += nbf(a)*nbf(b);
  }

  for (int i = 0; i < 7; ++i) {
    auto a = test::gaussian(C, K.second, pure);
    auto b = test::gaussian(D, 1, pure);
    auto r0 = test::random<double,3>(-0.25,0.25);
    auto r1 = test::random<double,3>(-0.25,0.25);
    ket.push_back({a,r0});
    ket.push_back({b,r1});
    kl.push_back(Index2{i*2,i*2+1});
    ncd += nbf(a)*nbf(b);
  }

  cuda::host::vector<double> result(nab*ncd);
  cudaStream_t stream = 0;
  auto eri = cuda::md::eri<4>(bra, ket, stream);
  eri->compute(ij, kl, result.data());
  cuda::stream::synchronize(stream);

  // std::cout << result[0] << std::endl;
  // std::cout << reference[0] << std::endl;

  auto *ab_cd = result.data();
#define ab_cd(ij,kl) ab_cd[(kl) + (ij)*(ncd)]

  for (auto [i,j] : ij) {
    auto &[a,ra] = bra[i];
    auto &[b,rb] = bra[j];
    for (auto [k,l] : kl) {
      //printf("(%i,%i,%i,%i)\n", i,j,k,l);
      auto &[c,rc] = ket[k];
      auto &[d,rd] = ket[l];
      auto ref = libintx::reference::eri(a,b,c,d);
      test::check4(
        [&](int ij, int kl, auto &&ref) {
          CHECK(ab_cd(ij,kl) == ref);
        },
        nbf(a), nbf(b), nbf(c), nbf(d),
        ref->compute(ra, rb, rc, rd)
        //ref->buffer()
      );
      ab_cd += nbf(c)*nbf(d);
    }
    ab_cd -= ncd;
    ab_cd += ncd*nbf(a)*nbf(b);
  }

#undef ab_cd

}


#define MD_ERI3_SUBCASE(A,B,X,Ks)                 \
  SUBCASE("A=" # A " B=" # B " X=" # X) {         \
    if (test::enabled(A,B,X)) {                   \
      for (auto K : Ks) {                         \
        md_eri3_test<A,B,X>(K);                   \
      }                                           \
    }                                             \
  }

#define MD_ERI4_SUBCASE(A,B,C,D,Ks)                     \
  SUBCASE("A=" # A " B=" # B " C=" # C " D=" # D) {     \
    if (test::enabled(A,B,C,D)) {                       \
      for (auto K : Ks) {                               \
        md_eri_test<A,B,C,D>(K);                        \
      }                                                 \
    }                                                   \
  }

TEST_CASE("cuda.md.eri4") {
  std::vector< std::pair<int,int> > Ks = {
    {1,1}, {5,1}, {5,5}
  };
  MD_ERI4_SUBCASE(0,0,0,0,Ks);
  MD_ERI4_SUBCASE(1,0,0,0,Ks);
  MD_ERI4_SUBCASE(0,0,1,0,Ks);
  MD_ERI4_SUBCASE(1,0,1,0,Ks);
  MD_ERI4_SUBCASE(1,1,1,1,Ks);
  MD_ERI4_SUBCASE(2,2,2,2,Ks);
  MD_ERI4_SUBCASE(2,0,1,1,Ks);
}

TEST_CASE("cuda.md.eri3") {

  return;

  std::vector< std::pair<int,int> > Ks = {
    {1,1} // , {5,1}, {5,5}
  };

  // MD_ERI3_SUBCASE(2,2,1,Ks);
  // // MD_ERI3_SUBCASE(1,0,0,Ks);
  // // MD_ERI3_SUBCASE(0,0,1,Ks);
  // //MD_ERI3_SUBCASE(2,0,2);
  // MD_ERI3_SUBCASE(1,1,1,Ks);
  // // MD_ERI3_SUBCASE(1,1,2);
  // // MD_ERI3_SUBCASE(3,0,0);
  // //MD_ERI3_SUBCASE(0,3,0);
  // MD_ERI3_SUBCASE(2,2,2,Ks);
  // MD_ERI3_SUBCASE(3,3,3,Ks);
  // MD_ERI3_SUBCASE(4,4,4,Ks);
  // MD_ERI3_SUBCASE(5,5,5,Ks);

  // MD_ERI3_SUBCASE(0,0,1);
  // MD_ERI3_SUBCASE(0,0,2);
  // MD_ERI3_SUBCASE(0,0,3);
  // MD_ERI3_SUBCASE(0,0,4);

  // MD_ERI3_SUBCASE(1,0,0);
  // MD_ERI3_SUBCASE(1,0,1);
  // MD_ERI3_SUBCASE(2,0,2);
  // MD_ERI3_SUBCASE(3,0,3);
  // MD_ERI3_SUBCASE(4,0,4);

  // MD_ERI3_SUBCASE(1,1,0);
  // MD_ERI3_SUBCASE(1,1,1);
  // MD_ERI3_SUBCASE(2,2,2);
  // MD_ERI3_SUBCASE(3,3,3);
  // MD_ERI3_SUBCASE(4,4,4);

}
