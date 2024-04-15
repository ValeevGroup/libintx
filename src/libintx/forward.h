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

  template<typename First, typename Second>
  struct pair {
    First first;
    Second second;
    // constexpr operator std::pair<First,Second>() const {
    //   return { first, second };
    // }
  };

  using Index1 = int;
  using Index2 = pair<Index1,Index1>;

  template<int Idx, typename First, typename Second>
  auto get(pair<First,Second>&& idx) {
    static_assert(Idx == 0 || Idx == 1);
    if constexpr (Idx == 0) return idx.first;
    if constexpr (Idx == 1) return idx.second;
  }

  template<int ... Args>
  struct IntegralEngine;

  // base class
  template<>
  struct IntegralEngine<> {
    virtual ~IntegralEngine() = default;
  };

  struct JEngine;

}

#endif /* LIBINTX_FORWARD_H */
