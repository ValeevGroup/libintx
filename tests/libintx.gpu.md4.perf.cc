#include "libintx/gpu/md/engine.h"
#include "libintx/gpu/api/api.h"
#include "libintx/utility.h"
#include "test.h"
#include "reference.h"
#include <iostream>

using namespace libintx;
using namespace libintx::gpu;
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
  size_t Nij, size_t Nkl)
{

  Basis<Gaussian> basis;

  std::array<size_t,2> dims{ Nij*npure(A)*npure(B), npure(C)*npure(D)*Nkl };
  auto buffer = device::vector<double>(dims[0]*dims[1]);

  printf("# (%i%i|%i%i) ", A, B, C, D);
  printf("dims=%ix%i, memory=%fGB\n", Nij, Nkl, 8*buffer.size()/1e9);

  struct {
    std::unique_ptr< libintx::IntegralEngine<2,2> > engine;
    double time = 0;
    std::vector<double> ratio;
  } md;

  for (auto K : Ks) {

    printf("# K={%i,%i}: ", K.first, K.second);

    auto [bra,ijs] = test::basis2({A,B}, {K.first,1}, Nij);
    auto [ket,kls] = test::basis2({C,D}, {K.second,1}, Nkl);

    gpuStream_t stream = 0;
    md.engine = libintx::gpu::md::eri<4>(bra, ket, stream);
    md.engine->max_memory = 2ul*1024*1024*1024;
    md.engine->compute(ijs, kls, buffer.data(), dims);
    libintx::gpu::stream::synchronize(stream);
    {
      auto t0 = time::now();
      md.engine->compute(ijs, kls, buffer.data(), dims);
      libintx::gpu::stream::synchronize(stream);
      double t = time::since(t0);
      md.time = 1/t;
    }

    printf("T(MD)=%f ", 1/md.time);
    printf("Int/s=%4.2e ", (Nij*Nkl)*md.time);

#ifdef LIBINTX_TEST_REFERENCE
    double tref = reference::time(Nij*Nkl, shell(bra[0]), shell(bra[1]), shell(ket[0]), shell(ket[1]));
    printf("T(Ref)=%f ", tref);
    printf("T(Ref/MD)=%f ", tref*md.time);
#endif

    printf("\n");

  } // Ks


}

#define RUN(A,B,C,D,...)             \
  if (test::enabled(A,B,C,D)) run(A,B,C,D,__VA_ARGS__);

int main(int argc, char **argv) {

  int load = 128;
  if (argc > 1) load = std::atoi(argv[1]);

  std::vector<Index2> Ks = {
    {1,1}, {1,5}, {5,5}
  };

  auto N = [&](int L) {
    int p = 1;
    for (int l = 1; l <= L; ++l) {
      p *= (l%2 ? 1 : 2);
    }
    return (8*load)/p;
  };

  // for (int a = 0; a <= LMAX; ++a) {
  //   for (int b = 0; b <= LMAX; ++b) {
  //     RUN(a,b,a,b, Ks, N(a), N(b));
  //   }
  // }

  std::vector<int> loadv = { load*128, load*16, load*8, load*4, load*2, load*1, load*1 };

  // (x,x,s,s)
  for (int l = 1; l <= LMAX; ++l) {
    RUN(l,l,0,0, Ks, 8*loadv.at(l), loadv.at(0));
  }

  // (x,s,x,s)
  for (int l = 1; l <= LMAX; ++l) {
    RUN(l,0,l,0, Ks, 4*loadv.at(l), 4*loadv.at(l));
  }

  // (x,x,x,x)
  for (int l = 0; l <= LMAX; ++l) {
    RUN(l,l,l,l, Ks, loadv.at(l), loadv.at(l));
  }

}
