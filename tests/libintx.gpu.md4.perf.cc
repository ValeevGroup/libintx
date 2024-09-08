#include "libintx/gpu/engine.h"
#include "libintx/gpu/api/api.h"
#include "libintx/utility.h"
#include "test.h"
#include "reference.h"
#include <iostream>

using namespace libintx;

auto run(
  Operator op,
  int A, int B, int C, int D,
  std::vector<Index2> Ks,
  size_t Nij, size_t Nkl)
{

  Basis<Gaussian> basis;

  std::array<size_t,2> dims{ Nij*npure(A)*npure(B), npure(C)*npure(D)*Nkl };
  auto buffer = gpu::device::vector<double>(dims[0]*dims[1]);

  printf("# (%i%i|%i%i) ", A, B, C, D);
  printf("dims=%ix%i, memory=%fGB\n", Nij, Nkl, 8*buffer.size()/1e9);

  struct {
    std::unique_ptr< libintx::gpu::IntegralEngine<4> > engine;
    double time = 0;
    std::vector<double> ratio;
  } md;

  for (auto K : Ks) {

    printf("# K={%i,%i}: ", K.first, K.second);

    auto [bra,ijs] = test::basis2({A,B}, {K.first,1}, Nij);
    auto [ket,kls] = test::basis2({C,D}, {K.second,1}, Nkl);

    gpuStream_t stream = 0;
    md.engine = libintx::gpu::integral_engine<4>(bra, ket, stream);
    md.engine->max_memory = 2ul*1024*1024*1024;
    md.engine->compute(Coulomb, ijs, kls, buffer.data(), dims);
    libintx::gpu::stream::synchronize(stream);
    {
      auto t0 = time::now();
      md.engine->compute(Coulomb, ijs, kls, buffer.data(), dims);
      libintx::gpu::stream::synchronize(stream);
      double t = time::since(t0);
      md.time = 1/t;
    }

    printf("T(MD)=%f ", 1/md.time);
    printf("Int/s=%4.2e ", (Nij*Nkl)*md.time);

#ifdef LIBINTX_TEST_REFERENCE
    double tref = reference::time(op, None, Nij*Nkl, bra[0], bra[1], ket[0], ket[1]);
    printf("T(Ref)=%f ", tref);
    printf("T(Ref/MD)=%f ", tref*md.time);
#endif

    printf("\n");

  } // Ks


}

#define RUN(OP,A,B,C,D,...)                                     \
  if (test::enabled(A,B,C,D)) run(OP,A,B,C,D,__VA_ARGS__);

int main(int argc, char **argv) {

  auto dims = test::parse_args<2>(argc,argv,6000);

  std::vector<Index2> Ks = {
    {1,1}, {1,5}, {5,5}
  };

  Operator op = Coulomb;

  // (x,x,x,x)
  for (int l = 0; l <= LMAX; ++l) {
    RUN(op, l,l,l,l, Ks, dims[0]/npure(l,l), dims[1]/npure(l,l));
  }

  // (x,x,s,s)
  for (int l = 1; l <= LMAX; ++l) {
    RUN(op, l,l,0,0, Ks, dims[0]/npure(l,l), dims[1]);
  }

  // (x,s,x,s)
  for (int l = 1; l <= LMAX; ++l) {
    RUN(op, l,0,l,0, Ks, dims[0]/npure(l), dims[1]/npure(l));
  }

}
