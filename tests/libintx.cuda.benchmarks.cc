#include "libintx/engine/os/engine.h"
#include "libintx/cuda/eri.h"
#include "libintx/utility.h"
#include "test.h"
#include <iostream>

using namespace libintx;
using namespace libintx::cuda;
using libintx::time;

const Double<3> r0 = {  0.7, -1.2, -0.1 };
const Double<3> r1 = { -1.0,  0.0,  0.3 };
const Double<3> rx = {  0.5, -1.5,  0.9 };

auto eri_test_case(int A, int B, int X, std::vector< std::array<int,2> > Ks = { {1,1} }, int N = 0) {

  printf("# %i%i%i\n", A, B, X);
  std::vector<double> ratios;

  for (auto K : Ks) {

  int Nk = N;
  if (!Nk) Nk = 20000000/(K[1]*npure(A+B)*npure(X));

  auto a = test::gaussian(A, K[0]);
  auto b = test::gaussian(B, K[1]);
  auto x = test::gaussian(X, 1);

  auto gpu = libintx::cuda::eri<3>();

  auto centers = std::vector< Double<3> >{r0,r1,rx};
  gpu->set_centers(centers);

  int AB = ncart(a)*ncart(b);
  int nbf = AB*npure(x);

  auto buffer = device::vector<double>(Nk*nbf);

  double tgpu = 0;
  for (int k = 0; k < 1; ++k) {
    IntegralList<3> list(Nk);
    for (int i = 0; i < list.size(); ++i) {
      list[i] = { {0,1,2}, buffer.data() + i*nbf };
    }
    auto t0 = time::now();
    gpu->compute(a, b, x, list);
    device::synchronize();
    double t = time::since(t0);
    tgpu = std::max(tgpu,1/t);
  }

  printf("# K=%i: ", K[0]*K[1]);
  printf("T(gpu)=%f ", 1/tgpu);

  double ratio = 0;
  {
    auto cpu = libintx::os::eri(a,b,x);
    auto t0 = time::now();
    for (int i = 0; i < Nk; ++i) {
      cpu->compute(r0, r1, rx);
    }
    auto t = time::since(t0);
    printf("T(cpu)=%f ", t);
    ratio = t*tgpu;
  }

  printf("T(cpu)/T(gpu)=%f\n", ratio);
  ratios.push_back(ratio);

  }

  printf("%i%i%i, ", A, B, X);
  for (auto r : ratios) {
    printf("%f, ", r);
  }
  printf("\n");

}

#define ERI_TEST_CASE(I,J,K,...)                                     \
  if (test::enabled(I,J,K)) { eri_test_case(I,J,K,__VA_ARGS__); }


int main() {

  std::vector< std::array<int,2> > K = { {1,1}, {5,1}, {5,5} };

  ERI_TEST_CASE(0,0,0, K);
  ERI_TEST_CASE(1,0,0, K);
  ERI_TEST_CASE(2,0,0, K);
  ERI_TEST_CASE(3,0,0, K);
  ERI_TEST_CASE(4,0,0, K);
  ERI_TEST_CASE(5,0,0, K);
  ERI_TEST_CASE(6,0,0, K);

  ERI_TEST_CASE(1,0,1, K);
  ERI_TEST_CASE(2,0,2, K);
  ERI_TEST_CASE(3,0,3, K);
  ERI_TEST_CASE(4,0,4, K);
  ERI_TEST_CASE(5,0,5, K);
  ERI_TEST_CASE(6,0,6, K);

  ERI_TEST_CASE(1,1,0, K);
  ERI_TEST_CASE(2,2,0, K);
  ERI_TEST_CASE(3,3,0, K);
  ERI_TEST_CASE(4,4,0, K);
  ERI_TEST_CASE(5,5,0, K);
  ERI_TEST_CASE(6,6,0, K);

  ERI_TEST_CASE(1,1,1, K);
  ERI_TEST_CASE(2,2,2, K);
  ERI_TEST_CASE(3,3,3, K);
  ERI_TEST_CASE(4,4,4, K);
  ERI_TEST_CASE(5,5,5, K);
  ERI_TEST_CASE(6,6,6, K);

}
