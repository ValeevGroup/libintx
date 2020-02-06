#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "boys/cuda/chebyshev.h"
#include <cuda/api_wrappers.hpp>
//#include <cuda/api/memory.hpp>

template<class Chebyshev>
__global__
void test(const Chebyshev chebyshev, double *ptr) {
  int bidx = blockIdx.x;
  int tidx = threadIdx.y + threadIdx.z*blockDim.y;
  ptr[bidx+tidx] = chebyshev.compute(bidx%128, tidx%8);
}

template<class Chebyshev>
__global__
void test_cooperative_groups(const Chebyshev chebyshev, double *ptr) {
  int bidx = blockIdx.x;
  int tidx = threadIdx.y + threadIdx.z*blockDim.y;
  auto p8 = cooperative_groups::tiled_partition<8>(cooperative_groups::this_thread_block());
  for (int i = 0; i < 8; ++i) {
    ptr[bidx+i] = chebyshev.compute(bidx%128, i, p8);
  }
}

template<int Order, int M, int Segments>
void test(int grid) {

  //dim3 block = { Order+1, 32/(Order+1), 4 };
  dim3 block = { 1, 8, 32 };

  typedef boys::cuda::Chebyshev<Order,M,117,Segments> Chebyshev;

  auto current_device = cuda::device::current::get();
  auto ptr = cuda::memory::device::make_unique<double[]>(current_device, grid*block.y*block.z);

  auto chebyshev = Chebyshev(current_device);

  for (size_t i = 0; i < 5; ++i) {
    test<<<grid,block>>>(chebyshev, ptr.get());
    current_device.synchronize();
    // test_cooperative_groups<<<grid,block>>>(chebyshev, ptr.get());
    // current_device.synchronize();
  }

  // cuda::launch(
  //   test<Chebyshev>,
  //   { grid, block },
  //   chebyshev,
  //   ptr.get()
  // );

  current_device.synchronize();

}

TEST_CASE("chebyshev") { test<7,20,117*7>(100*128); }
TEST_CASE("chebyshev") { test<7,40,117*7>(100*128); }
TEST_CASE("chebyshev") { test<15,20,117*7>(100*128); }
TEST_CASE("chebyshev") { test<15,40,117*7>(100*128); }
