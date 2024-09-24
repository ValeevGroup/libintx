#include "libintx/ao/md/engine.h"
#include "libintx/utility.h"
#include "test.h"
#include "reference.h"

using namespace libintx;

constexpr int repeat = 4;

auto run(Operator op, int A, int B, int C, int D, Index2 K, int Nab, int Ncd) {

  using IntegralEngine = libintx::md::IntegralEngine<4>;

  printf("- (%i%i|%i%i):", A, B, C, D);
  printf(" Dims=%ix%i ", Nab, Ncd);

  struct {
    std::unique_ptr<IntegralEngine> engine;
    double time = 0;
    std::vector<double> ratio;
  } md;

  auto [bra,ijs] = test::make_basis<2>({A,B}, {K.first,1}, Nab);
  auto [ket,kls] = test::make_basis<2>({C,D}, {K.second,1}, Ncd);

  //std::fill(buffer.begin(), buffer.end(), 0);
  md.engine = std::make_unique<IntegralEngine>(
    std::make_shared< Basis<Gaussian> >(bra),
    std::make_shared< Basis<Gaussian> >(ket)
  );
  md.engine->num_threads = 1;

  for (size_t i = 0; i < repeat; ++i) {
    auto t0 = time::now();
    IntegralEngine::Visitor V = [&](auto...) {
      // dont write results out
    };
    md.engine->compute(op, ijs, kls, {}, V);
    double t = time::since(t0);
    md.time = std::max(md.time,1/t);
  }

  printf("Int/s=%4.2e ", (Nab*Ncd)*md.time);
  // printf("T1(us)=%f ", (1/md.time)/(Nab*Nx)*1e6);
  printf("T(MD)=%f ", 1/md.time);

  double tref = 0;

#ifdef LIBINTX_TEST_REFERENCE
  for (size_t i = 0; i < repeat; ++i) {
    auto t = reference::time(op, None, Nab*Ncd, bra[0], bra[1], ket[0], ket[1]);
    tref = std::max(1/t,tref);
  }
#endif

  if (tref) {
    //printf("Ints/s(Ref)=%4.2e ", (Nab*Nx)/tref);
    printf("T(Ref)=%f ", 1/tref);
    printf("T(Ref/MD)=%.2f ", (1/tref)*md.time);
  }

  printf("\n");

}

#define RUN(OP,A,B,C,D,...)                                     \
  if (test::enabled(A,B,C,D)) run(OP,A,B,C,D,__VA_ARGS__);

int main(int argc, char **argv) {

  setenv("OMP_NUM_THREADS", "1", 0);
  setenv("MKL_NUM_THREADS", "1", 0);

  printf("# 4-center performance test\n");
  printf("%s", test::header().c_str());

  auto dims = test::parse_args<2>(argc,argv,4000);

  std::vector<Index2> Ks = {
    {1,1}, {1,10}, {5,10}
  };

  auto Ops = std::vector<Operator>{ Coulomb };
  for (auto op :  Ops) {
    for (auto K : Ks) {
      printf("---\n");
      printf("K: [%i,%i]\n", K.first, K.second);
      printf("results:\n");
      for (int a = 0; a <= LMAX; ++a) {
        for (int b = 0; b <= a; ++b) {
          for (int c = 0; c <= LMAX; ++c) {
            for (int d = 0; d <= c; ++d) {
              if (npure(a,b) < npure(c,d)) continue;
              if (std::max(a,b) < std::max(c,d)) continue;
              // sample
              if (a+b+c+d <= 1) goto run;
              if (a != b || c != d) continue;
              if (1 < c && c < a) continue;
            run:
              RUN(op,a,b,c,d,K, dims[0]/npure(a,b), dims[1]/npure(c,d));
            }
          }
        }
      }
    }
  }

}
