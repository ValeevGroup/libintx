#include "libintx/cuda/md/engine.h"
#include "libintx/cuda/api/api.h"
#include "libintx/utility.h"
#include "test.h"
#include <iostream>

#include "libintx/reference.h"

using namespace libintx;
using namespace libintx::cuda;
using libintx::time;

const Double<3> rs[] = {
  {  0.7, -1.2, -0.1 },
  { -1.0,  0.0,  0.3 },
  { -1.0,  3.0,  1.3 },
  {  4.0,  1.0, -0.7 }
};

template<typename ... Args>
double reference(int N, Args ... args) {
  auto eri = libintx::reference::eri(std::get<0>(args)...);
  auto t = time::now();
  eri->repeat(N, std::get<1>(args)...);
  return time::since(t);
}

auto eri4_test_case(
  int A, int B, int C, int D,
  std::vector< std::array<int,2> > Ks,
  int Nij, int Nkl)
{

  Basis<Gaussian> basis;

  auto buffer = device::vector<double>(npure(A)*npure(B)*Nij*npure(C)*npure(D)*Nkl);

  printf("# (%i%i|%i%i) ", A, B, C, D);
  printf("dims: %ix%i, memory=%f GB\n", Nij, Nkl, 8*buffer.size()/1e9);

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
    Basis<Gaussian> bra, ket;
    bra.push_back({a,rs[0]});
    for (int i = 0; i < Nij; ++i) {
      bra.push_back({b,rs[1]});
    }
    ket.push_back({c,rs[2]});
    for (int i = 0; i < Nkl; ++i) {
      ket.push_back({d,rs[3]});
    }
    //Basis<Gaussian> ket = { {c,ra}, {d,rb} };

    cudaStream_t stream = 0;
    md.engine = libintx::cuda::md::eri<4>(bra, ket, stream);
    std::vector<Index2> ab, cd;
    //std::vector<Index2> cd(Nkl, Index2{0,1});
    for (int i = 0; i < Nij; ++i) {
      ab.push_back(Index2{0,i+1});
    }
    for (int i = 0; i < Nkl; ++i) {
      cd.push_back(Index2{0,i+1});
    }
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
    double tref = ::reference(Nij*Nkl, bra[0], bra[1], ket[0], ket[1]);
    printf("T(Ref)=%f ", tref);
    printf("T(Ref/MD)=%f ", tref*md.time);
    printf("\n");

  } // Ks

}

#define ERI4_TEST_CASE(A,B,C,D,...)             \
  if (test::enabled(A,B,C,D)) eri4_test_case(A,B,C,D,__VA_ARGS__);

int main() {

  std::vector< std::array<int,2> > K = {
    {1,1}, {1,5}, {5,5}
  };

  ERI4_TEST_CASE(2,0,2,0, K, 1000, 1000);
  ERI4_TEST_CASE(2,2,2,0, K, 512, 512);
  ERI4_TEST_CASE(2,2,2,2, K, 512/2, 512/2);
  ERI4_TEST_CASE(3,3,3,3, K, 512/2, 512/2);
  ERI4_TEST_CASE(4,4,4,4, K, 512/4, 512/4);
  ERI4_TEST_CASE(5,5,5,5, K, 512/4, 512/4);
  ERI4_TEST_CASE(6,6,6,6, K, 512/4, 512/4);

}
