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

auto eri_test_case(int A, int B, int X, std::vector<int> Ks = { 1 }, int N = 0) {

  if (!N) N = 1000000/(1+A*B*X);

  printf("# %i%i%i, N=%i:\n", A, B, X, N);
  std::vector<double> ratios;

  for (auto K : Ks) {

  auto a = test::gaussian(A, K);
  auto b = test::gaussian(B, 1);
  auto x = test::gaussian(X, 1);

  auto gpu = libintx::cuda::eri<3>();

  auto centers = std::vector< Double<3> >{r0,r1,rx};
  gpu->set_centers(centers);

  int AB = ncart(a)*ncart(b);
  int nbf = AB*npure(x);

  auto buffer = device::vector<double>(N*nbf);

  IntegralList<3> list(N);
  for (int i = 0; i < N; ++i) {
    list[i] = { {0,1,2}, buffer.data() + i*nbf };
  }

  double tgpu = 0;
  for (int k = 0; k < 2; ++k) {
    auto t0 = time::now();
    gpu->compute(a, b, x, list);
    device::synchronize();
    double t = time::since(t0);
    tgpu = std::max(tgpu,1/t);
  }

  printf("# K=%i: ", K);
  printf("T(gpu)=%f ", 1/tgpu);

  double ratio = 0;
  {
    auto cpu = libintx::os::eri(a,b,x);
    auto t0 = time::now();
    for (int i = 0; i < N; ++i) {
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

  std::vector<int> K = { 1, 5, 10 };

  ERI_TEST_CASE(0,0,0, K);
  ERI_TEST_CASE(1,0,0, K);
  ERI_TEST_CASE(2,0,0, K);
  ERI_TEST_CASE(3,0,0, K);
  ERI_TEST_CASE(4,0,0, K);
  ERI_TEST_CASE(5,0,0, K);
  ERI_TEST_CASE(6,0,0, K);

  ERI_TEST_CASE(3,0,1, K);
  ERI_TEST_CASE(3,0,2, K);
  ERI_TEST_CASE(3,0,3, K);
  ERI_TEST_CASE(3,0,4, K);
  ERI_TEST_CASE(3,0,5, K);
  ERI_TEST_CASE(3,0,6, K);

  ERI_TEST_CASE(6,1,0, K);
  ERI_TEST_CASE(6,2,0, K);
  ERI_TEST_CASE(6,3,0, K);
  ERI_TEST_CASE(6,4,0, K);
  ERI_TEST_CASE(6,5,0, K);
  ERI_TEST_CASE(6,6,0, K);

  ERI_TEST_CASE(1,0,1, K);
  ERI_TEST_CASE(1,1,1, K);
  ERI_TEST_CASE(2,1,2, K);
  ERI_TEST_CASE(2,2,2, K);
  ERI_TEST_CASE(3,2,3, K);
  ERI_TEST_CASE(3,3,3, K);
  ERI_TEST_CASE(4,3,4, K);
  ERI_TEST_CASE(4,4,4, K);
  ERI_TEST_CASE(5,4,5, K);
  ERI_TEST_CASE(5,5,5, K);
  ERI_TEST_CASE(6,5,6, K);
  ERI_TEST_CASE(6,6,6, K);

}
