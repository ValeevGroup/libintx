#include "libintx/ao/md/engine.h"
#include "libintx/utility.h"
#include "test.h"
#include "reference.h"

using namespace libintx;
constexpr int ntries = 3;

auto run(Operator op, int A, int B, Index2 K, int Na, int Nb) {

  using IntegralEngine = libintx::md::IntegralEngine<2>;

  std::array<size_t,3> dims{ (size_t)Na, (size_t)Nb, (size_t)npure(A,B) };
  auto buffer = std::vector<double>(dims[0]*dims[1]*dims[2]);

  printf("- (%i|%i): ", A, B);
  printf(" Dims=%ix%i Mem(GB)=%.2f ", Na, Nb, 8*buffer.size()/1e9);

  struct {
    std::unique_ptr<IntegralEngine> engine;
    double time = 0;
    std::vector<double> ratio;
  } md;

  std::vector<double> csv[2];

  std::vector< std::tuple<int,std::array<double,3> > > params(100);
  for (auto &[Z,r] : params) {
    Z = test::random<int>(1,100);
    r = test::random<double,3>(-10,10);
  }

  auto [bra,is] = test::make_basis<1>({A}, {K.first}, Na);
  auto [ket,js] = test::make_basis<1>({B}, {K.second}, Nb);

  std::vector<Index2> ijs(Na*Nb);
  for (auto j : js) {
    for (auto i : is) {
      ijs[i] = {i,j};
    }
  }

  std::fill(buffer.begin(), buffer.end(), 0);
  md.engine = std::make_unique<IntegralEngine>(bra, ket);
  md.engine->num_threads = 1;
  md.engine->set(Nuclear::Operator::Parameters{ params });

  for (int i = 0; i < ntries; ++i) {
    (void)i;
    auto t0 = time::now();
    auto V = [&](size_t batch, size_t idx, const double *V, size_t ldV) {
      // const auto& [i,j] = *ijs;
      // int ij = i;
      // //printf("N=%i,i=%i,j=%i\n", N, i, j);
      // for (int iab = 0; iab < npure(A,B); ++iab) {
      //   int idx = ij + iab*Na*Nb;
      //   std::copy_n(V + iab*N, N, buffer.data()+idx);
      // }
    };
    md.engine->compute(op, ijs, V);
    double t = time::since(t0);
    md.time = std::max(1/t, md.time);
  }

  //printf("T(1)=%f (us) ", (1/md.time)/(Na*Nb)*1e6);
  printf("Int/s=%4.2e ", (Na*Nb)*md.time);
  printf("T(MD)=%f ", 1/md.time);

  double tref = 0;
#ifdef LIBINTX_TEST_REFERENCE
  for (int i = 0; i < ntries; ++i) {
    (void)i;
    auto t = reference::time(op, params, ijs.size(), bra[0], ket[0]);
    tref = std::max(1/t, tref);
  }
#endif

  if (tref) {
    //printf("Ints/s(Ref)=%4.2e ", (Nab*Nx)/tref);
    printf("T(Ref)=%f ", 1/tref);
    printf("T(Ref/MD)=%.2f ", md.time/tref);
  }

  printf("\n");

}

#define RUN(OP,A,B,...)                         \
  if (test::enabled(A,B)) run(OP,A,B,__VA_ARGS__);

int main(int argc, char **argv) {

  setenv("OMP_NUM_THREADS", "1", 0);
  setenv("MKL_NUM_THREADS", "1", 0);

  printf("\n");
  printf("# 2-center performance test\n");
  printf("%s", test::header().c_str());

  auto dims = test::parse_args<2>(argc,argv,500);

  std::vector<Index2> Ks = {
    {1,1}, {1,5} //, {1,9}
  };

  auto Ops = std::vector<Operator>{
    // Overlap,
    // Kinetic,
    Nuclear
  };
  //auto Ops = std::vector<Operator>{ Nuclear };

  for (auto op :  Ops) {
    for (auto K : Ks) {
      printf("---\n");
      printf("Operator: %i\n", (int)op);
      printf("K: [%i,%i]\n", K.first, K.second);
      printf("results:\n");
      for (int b = 0; b <= LMAX; ++b) {
        for (int a = 0; a <= LMAX; ++a) {
          if (a < b) continue;
          if (a%std::max(b,1)) continue;
          RUN(op, a, b, K, dims[0]/npure(a), dims[1]/npure(b));
        }
      }
    }
  }

}
