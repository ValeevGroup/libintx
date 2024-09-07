#include "libintx/gpu/md/engine.h"
#include "libintx/gpu/api/api.h"
#include "libintx/utility.h"
#include "test.h"
#include "reference.h"
#include <iostream>

using namespace libintx;

auto run(
  int X, int C, int D,
  std::vector<Index2> Ks,
  int Nij, int Nkl)
{

  Basis<Gaussian> basis;

  std::array<size_t,2> dims{ Nij*npure(X), npure(C)*npure(D)*Nkl };
  auto buffer = gpu::device::vector<double>(dims[0]*dims[1]);

  printf("# (%i|%i%i) ", X, C, D);
  printf("dims: %ix%i, memory=%f GB\n", Nij, Nkl, 8*buffer.size()/1e9);

  struct {
    std::unique_ptr< libintx::IntegralEngine<1,2> > engine;
    double time = 0;
    std::vector<double> ratio;
  } md;

  for (auto K : Ks) {

    printf("# K={%i,%i}: ", K.first, K.second);

    auto [bra,is] = test::basis1({X}, {K.first}, Nij);
    auto [ket,kls] = test::basis2({C,D}, {K.second,1}, Nkl);

    gpuStream_t stream = 0;
    md.engine = libintx::gpu::md::eri<3>(bra, ket, stream);
    md.engine->compute(is, kls, buffer.data(), dims);
    libintx::gpu::stream::synchronize(stream);
    {
      auto t0 = time::now();
      md.engine->compute(is, kls, buffer.data(), dims);
      libintx::gpu::stream::synchronize(stream);
      double t = time::since(t0);
      md.time = 1/t;
    }

    printf("T(MD)=%f ", 1/md.time);
    printf("Int/s=%4.2e ", (Nij*Nkl)*md.time);

#ifdef LIBINTX_TEST_REFERENCE
    double tref = reference::time(Nij*Nkl, shell(bra[0]), shell(ket[0]), shell(ket[1]));
    printf("T(Ref)=%f ", tref);
    printf("T(Ref/MD)=%f ", tref*md.time);
#endif

    printf("\n");

  } // Ks

}

#define RUN(X,C,D,...)                                  \
  if (test::enabled(X,C,D)) run(X,C,D,__VA_ARGS__);

int main(int argc, char **argv) {

  auto dims = test::parse_args<2>(argc,argv,6000);

  std::vector<Index2> Ks = {
    {1,1}, {1,5}, {5,5}
  };

  for (int l = 0; l <= LMAX; ++l) {
    RUN(l,l,l,Ks, dims[0]/npure(l), dims[1]/npure(l,l));
  }

  for (int x = 1; x <= LMAX; ++x) {
    RUN(x,0,0,Ks, dims[0]/npure(x), dims[1]);
  }

  for (int l = 1; l <= LMAX; ++l) {
    RUN(0,l,l,Ks, dims[0], dims[1]/npure(l,l));
  }

}
