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

auto run(
  int A, int B, int C, int D,
  std::vector<Index2> Ks,
  int Nij, int Nkl)
{

  Basis<Gaussian> basis;

  std::array<size_t,2> dims{ Nij*npure(A)*npure(B), npure(C)*npure(D)*Nkl };
  auto buffer = device::vector<double>(dims[0]*dims[1]);

  printf("# (%i%i|%i%i) ", A, B, C, D);
  printf("dims: %ix%i, memory=%f GB\n", Nij, Nkl, 8*buffer.size()/1e9);

  struct {
    std::unique_ptr< libintx::IntegralEngine<4> > engine;
    double time = 0;
    std::vector<double> ratio;
  } md;

  for (auto K : Ks) {

    printf("# K={%i,%i}: ", K.first, K.second);

    auto [bra,ijs] = test::basis2({A,B}, {K.first,1}, Nij);
    auto [ket,kls] = test::basis2({C,D}, {K.second,1}, Nkl);

    cudaStream_t stream = 0;
    md.engine = libintx::cuda::md::eri<4>(bra, ket, stream);
    md.engine->compute(ijs, kls, buffer.data(), dims);
    libintx::cuda::stream::synchronize(stream);
    {
      auto t0 = time::now();
      md.engine->compute(ijs, kls, buffer.data(), dims);
      libintx::cuda::stream::synchronize(stream);
      double t = time::since(t0);
      md.time = 1/t;
    }

    printf("T(MD)=%f ", 1/md.time);
    double tref = reference::time(Nij*Nkl, bra[0], bra[1], ket[0], ket[1]);
    printf("T(Ref)=%f ", tref);
    printf("T(Ref/MD)=%f ", tref*md.time);
    printf("\n");

  } // Ks

}

#define RUN(A,B,C,D,...)             \
  if (test::enabled(A,B,C,D)) run(A,B,C,D,__VA_ARGS__);

int main() {

  std::vector<Index2> Ks = {
    {1,1}, {1,5}, {5,5}
  };

  auto N = [](int L) {
    int p = 1;
    for (int l = 1; l <= L; ++l) {
      p *= (l%2 ? 1 : 2);
    }
    return (2*1024)/p;
  };

  // for (int a = 0; a <= LMAX; ++a) {
  //   for (int b = 0; b <= LMAX; ++b) {
  //     RUN(a,b,a,b, Ks, N(a), N(b));
  //   }
  // }

  RUN(2,0,2,0, Ks, 1000, 1000);
  RUN(2,2,2,0, Ks, 512, 512);
  RUN(2,2,2,2, Ks, 512/2, 512/2);
  RUN(3,3,3,3, Ks, 512/2, 512/2);
  RUN(4,4,4,4, Ks, 512/4, 512/4);
  RUN(5,5,5,5, Ks, 512/4, 512/4);
  RUN(6,6,6,6, Ks, 512/4, 512/4);

}
