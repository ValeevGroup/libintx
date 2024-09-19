#include "libintx/ao/md/engine.h"
#include "libintx/utility.h"
#include "libintx/simd.h"
#include "libintx/blas.h"
#include "test.h"
#include "reference.h"

using namespace libintx;

constexpr int ntries = 3;

auto run(Operator op, int X, int A, int B, BraKet<int> K, int Nx, int Nab) {

  using IntegralEngine = libintx::md::IntegralEngine<3>;

  // std::array<size_t,3> dims{ (size_t)Nx, (size_t)Nab, (size_t)npure(A,B)*npure(X) };
  // auto buffer = std::vector<double>(dims[0]*dims[1]*dims[2]);

  printf("- (%i|%i%i):", X, A, B);
  printf(" Dims=%ix%i Mem(GB)=%.2f ", Nx, Nab, (1/1e9)*Nx*Nab*npure(X)*npure(A,B)*8);

  struct {
    std::unique_ptr<IntegralEngine> engine;
    double time = 0;
    std::vector<double> ratio;
  } md;

  auto [aux,xs] = test::make_basis<1>({X}, {K.bra}, Nx);
  auto [bra,ijs] = test::make_basis<2>({A,B}, {K.ket,1}, Nab);

  //std::fill(buffer.begin(), buffer.end(), 0);
  md.engine = std::make_unique<IntegralEngine>(
    std::make_shared< Basis<Gaussian> >(aux),
    std::make_shared< Basis<Gaussian> >(bra)
  );
  md.engine->num_threads = 1;

  for (int i = 0; i < ntries; ++i) {
    auto t0 = time::now();
    IntegralEngine::Visitor V = [&](auto ...) {
      // dont write results out
    };
    md.engine->compute(op, xs, ijs, {}, V);
    double t = time::since(t0);
    md.time = std::max(md.time, 1/t);
  }

  printf("Int/s=%4.2e ", (Nab*Nx)*md.time);
  // printf("T1(us)=%f ", (1/md.time)/(Nab*Nx)*1e6);
   printf("T(MD)=%f ", 1/md.time);

  double tref = 0;

#ifdef LIBINTX_TEST_REFERENCE
  for (int i = 0; i < ntries; ++i) {
    auto t = reference::time(op, None, Nx*Nab, aux[0], bra[0], bra[1]);
    tref = std::max(1/t,tref);
  }
#endif

  if (tref) {
    //printf("Ints/s(Ref)=%4.2e ", (Nab*Nx)/tref);
    printf("T(Ref)=%f ", 1/tref);
    printf("T(Ref/MD)=%.2f ", md.time/tref);
  }

  printf("\n");

} // Ks

#define RUN(OP,X,A,B,...)                               \
  if (test::enabled(X,A,B)) run(OP,X,A,B,__VA_ARGS__);

int main(int argc, char **argv) {

  setenv("OMP_NUM_THREADS", "1", 0);
  setenv("MKL_NUM_THREADS", "1", 0);

  printf("\n");

  printf("# 3-center performance test\n");
  printf("%s", test::header().c_str());
  printf("%s\n", blas::version().c_str());

  auto dims = test::parse_args<2>(argc,argv,2000);

  std::vector< BraKet<int> > Ks = {
    {1,1}, {1,10}, {5,10}
  };

  auto Ops = std::vector<Operator>{ Coulomb };

  for (auto op :  Ops) {
    for (auto K : Ks) {
      printf("---\n");
      printf("K: [%i,%i]\n", K.bra, K.ket);
      printf("results:\n");
      for (int a = 0; a <= LMAX; ++a) {
        for (int b = 0; b <= a; ++b) {
          for (int x = 0; x <= XMAX; ++x) {
            if (a != b) continue;
            if (x == a || x == 1) goto run;
            if (a < 2 && x%2 == 0) goto run;
            continue;
          run:
            RUN(op,x,a,b,K, dims[0]/npure(x), dims[1]/npure(a,b));
          }
        }
      }
    }
  }

}
