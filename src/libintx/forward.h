#ifndef LIBINTX_FORWARD_H
#define LIBINTX_FORWARD_H

#define LIBINTX_NOINLINE __attribute__((noinline))

#ifdef __CUDACC__
#define LIBINTX_GPU_DEVICE __device__
#define LIBINTX_GPU_ENABLED __host__ __device__
#define LIBINTX_GPU_FORCEINLINE __forceinline__
#define LIBINTX_GPU_CONSTANT __constant__
#else
#define LIBINTX_GPU_DEVICE
#define LIBINTX_GPU_ENABLED
#define LIBINTX_GPU_FORCEINLINE inline
#define LIBINTX_GPU_CONSTANT
#endif

namespace boys {

  template<int Order, int M, int MaxT, int Segments>
  struct Chebyshev;

}

namespace libintx {

  template<typename T, int N>
  struct array;

  struct Shell;
  struct Gaussian;

  struct Index2 {
    int first, second;
  };

  template<int Centers, int Electrons = 2>
  struct IntegralEngine;

  struct JEngine;

}

#endif /* LIBINTX_FORWARD_H */
