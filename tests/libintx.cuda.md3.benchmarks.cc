#include "libintx/cuda/md/engine.h"
#include "libintx/cuda/api/api.h"
#include "libintx/utility.h"
#include "test.h"
#include <iostream>

#include "libintx/reference.h"

using namespace libintx;

auto run(
  int X, int C, int D,
  std::vector<Index2> Ks,
  int Nij, int Nkl)
{

  Basis<Gaussian> basis;

  std::array<size_t,2> dims{ Nij*npure(X), npure(C)*npure(D)*Nkl };
  auto buffer = cuda::device::vector<double>(dims[0]*dims[1]);

  printf("# (%i|%i%i) ", X, C, D);
  printf("dims: %ix%i, memory=%f GB\n", Nij, Nkl, 8*buffer.size()/1e9);

  struct {
    std::unique_ptr< libintx::IntegralEngine<3> > engine;
    double time = 0;
    std::vector<double> ratio;
  } md;

  for (auto K : Ks) {

    printf("# K={%i,%i}: ", K.first, K.second);

    auto [bra,is] = test::basis1({X}, {K.first}, Nij);
    auto [ket,kls] = test::basis2({C,D}, {K.second,1}, Nkl);

    cudaStream_t stream = 0;
    md.engine = libintx::cuda::md::eri<3>(bra, ket, stream);
    md.engine->compute(is, kls, buffer.data(), dims);
    libintx::cuda::stream::synchronize(stream);
    {
      auto t0 = time::now();
      md.engine->compute(is, kls, buffer.data(), dims);
      libintx::cuda::stream::synchronize(stream);
      double t = time::since(t0);
      md.time = 1/t;
    }

    printf("T(MD)=%f ", 1/md.time);
    printf("Int/s=%4.2e ", (Nij*Nkl)*md.time);
    double tref = reference::time(Nij*Nkl, bra[0], ket[0], ket[1]);
    printf("T(Ref)=%f ", tref);
    printf("T(Ref/MD)=%f ", tref*md.time);
    printf("\n");

  } // Ks

}

#define RUN(X,C,D,...)                                  \
  if (test::enabled(X,C,D)) run(X,C,D,__VA_ARGS__);

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

  for (int x = 0; x <= XMAX; ++x) {
    for (int c = 0; c <= LMAX; ++c) {
      for (int d = 0; d <= c; ++d) {
        RUN(x,c,d,Ks, 2*N(x), N(c));
      }
    }
  }

}
