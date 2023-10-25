#include "libintx/cuda/md/engine.h"
#include "libintx/utility.h"
#include "test.h"
#include <iostream>

#include "libintx/reference.h"

using namespace libintx;
using namespace libintx::cuda;
using libintx::time;

const Double<3> ra = {  0.7, -1.2, -0.1 };
const Double<3> rb = { -1.0,  0.0,  0.3 };

template<typename ... Args>
double reference(int N, Args ... args) {
  auto eri = libintx::reference::eri(std::get<0>(args)...);
  auto t = time::now();
  for (int i = 0; i < N; ++i) {
    eri->compute(std::get<1>(args)...);
  }
  return time::since(t);
}

auto eri4_test_case(int A, int B, int C, int D, std::vector< std::array<int,2> > Ks = { {1,1} }, int N = 0) {

  int nab = 200/(A*B+1);
  int ncd = 32*10;
  if (!N) N = nab*ncd;

  Basis<Gaussian> basis;

  int AB = npure(A)*npure(B);
  int CD = npure(C)*npure(D);
  int nbf = AB*CD;
  auto buffer = device::vector<double>(N*nbf);
  printf("# (%i%i|%i%i) ", A, B, C, D);
  printf("dims: %ix%i, memory=%f GB\n", nab, ncd, 8*buffer.size()/1e9);

  struct {
    std::unique_ptr< libintx::IntegralEngine<4> > engine;
    double time = 0;
    std::vector<double> ratio;
  } md;

  for (auto K : Ks) {

    printf("# K={%i,%i}: ", K[0], K[1]);

    auto a = test::gaussian(A, K[0]);
    auto b = test::gaussian(B, 1);
    auto c = test::gaussian(C, K[1]);
    auto d = test::gaussian(D, 1);
    Basis<Gaussian> bra = { {a,ra}, {b,rb} };
    Basis<Gaussian> ket = { {c,ra}, {d,rb} };

    double tref = ::reference(N, bra[0], bra[1], ket[0], ket[1]);
    printf("T(Ref)=%f ", tref);

    cudaStream_t stream = 0;
    md.engine = libintx::cuda::md::eri<4>(bra, ket, stream);
    std::vector<Index2> ab(nab, Index2{0,1});
    std::vector<Index2> cd(ncd, Index2{0,1});
    md.engine->compute(ab, cd, buffer.data());
    libintx::cuda::stream::synchronize(stream);
    {
      auto t0 = time::now();
      md.engine->compute(ab, cd, buffer.data());
      libintx::cuda::stream::synchronize(stream);
      double t = time::since(t0);
      md.time = 1/t;
    }

    printf("T(MD)=%f ", 1/md.time);
    printf("T(Ref/MD)=%f ", tref*md.time);
    printf("\n");

  } // Ks

}


#define ERI_TEST_CASE(I,J,K,...)                                     \
  if (test::enabled(I,J,K)) { eri_test_case(I,J,K,__VA_ARGS__); }

#define ERI4_TEST_CASE(A,B,C,D,...)                                     \
  if (test::enabled(A,B,C,D)) { eri4_test_case(A,B,C,D,__VA_ARGS__); }


int main() {

  std::vector< std::array<int,2> > K = {
    {1,1}, {1,5}, {5,5}
  };

  ERI4_TEST_CASE(2,0,2,0, K);
  ERI4_TEST_CASE(2,2,2,0, K);
  ERI4_TEST_CASE(2,2,2,2, K);
  ERI4_TEST_CASE(3,3,3,3, K);
  ERI4_TEST_CASE(4,4,4,4, K);
  ERI4_TEST_CASE(5,5,5,5, K);
  ERI4_TEST_CASE(6,6,6,6, K);

  //return 0;

  //ERI_TEST_CASE(2,2,0, K);
  // ERI_TEST_CASE(0,0,0, K);
  // ERI_TEST_CASE(1,0,0, K);
  // ERI_TEST_CASE(1,1,0, K);
  // ERI_TEST_CASE(2,0,0, K);
  // ERI_TEST_CASE(2,1,0, K);
  // ERI_TEST_CASE(2,2,0, K);
  // ERI_TEST_CASE(2,2,0, K);
  // ERI_TEST_CASE(3,1,0, K);
  // ERI_TEST_CASE(3,2,0, K);

  // ERI_TEST_CASE(6,6,6, K);
  // ERI_TEST_CASE(5,5,5, K);
  // ERI_TEST_CASE(4,4,4, K);
  // ERI_TEST_CASE(3,3,3, K);
  // ERI_TEST_CASE(2,2,2, K);
  // ERI_TEST_CASE(1,1,1, K);
  // ERI_TEST_CASE(0,0,0, K);

  //ERI_TEST_CASE(4,3,0, K);

  // ERI_TEST_CASE(0,0,0, K);
  // ERI_TEST_CASE(1,0,0, K);
  // ERI_TEST_CASE(2,0,0, K);
  // ERI_TEST_CASE(3,0,0, K);
  // ERI_TEST_CASE(4,0,0, K);
  // ERI_TEST_CASE(5,0,0, K);
  // ERI_TEST_CASE(6,0,0, K);

  // ERI_TEST_CASE(3,0,1, K);
  // ERI_TEST_CASE(3,0,2, K);
  // ERI_TEST_CASE(3,0,3, K);
  // ERI_TEST_CASE(3,0,4, K);
  // ERI_TEST_CASE(3,0,5, K);
  // ERI_TEST_CASE(3,0,6, K);

  // ERI_TEST_CASE(6,1,0, K);
  // ERI_TEST_CASE(6,2,0, K);
  // ERI_TEST_CASE(6,3,0, K);
  // ERI_TEST_CASE(6,4,0, K);
  // ERI_TEST_CASE(6,5,0, K);
  // ERI_TEST_CASE(6,6,0, K);

  // ERI_TEST_CASE(1,0,1, K);
  // ERI_TEST_CASE(1,1,1, K);
  // ERI_TEST_CASE(2,1,2, K);
  // ERI_TEST_CASE(2,2,2, K);
  // ERI_TEST_CASE(3,2,3, K);
  // ERI_TEST_CASE(3,3,3, K);
  // ERI_TEST_CASE(4,3,4, K);
  // ERI_TEST_CASE(4,4,4, K);
  // ERI_TEST_CASE(5,4,5, K);
  // ERI_TEST_CASE(5,5,5, K);
  // ERI_TEST_CASE(6,5,6, K);
  // ERI_TEST_CASE(6,6,6, K);

}
