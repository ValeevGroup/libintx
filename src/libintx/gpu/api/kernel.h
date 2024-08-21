#ifndef LIBINTX_GPU_API_KERNEL_H
#define LIBINTX_GPU_API_KERNEL_H

namespace libintx::gpu::kernel {

  struct launch_bounds {
    int max_threads, min_blocks;
  };

  template<class F, class ... Args>
  __global__
  void launch(F f, Args ... args) {
    f(args...);
  }

  template<int maxThreads, int minBlocks, class F, class ... Args>
  __global__ __launch_bounds__(maxThreads,minBlocks)
  void launch(F f, Args ... args) {
    f(args...);
  }

}

#endif /* LIBINTX_GPU_API_KERNEL_H */
