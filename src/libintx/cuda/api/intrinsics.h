#ifndef LIBINTX_CUDA_API_INTRINSICS_H
#define LIBINTX_CUDA_API_INTRINSICS_H

namespace libintx::cuda {

  __device__
  T ldg(const T *ptr) {
    static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
      using U = uint32_t;
      U v = __ldg(reinterpret_cast<const U*>(ptr));
      return *reinterpret_cast<const T*>(&v);
    }
    if constexpr (sizeof(T) == sizeof(uint64_t)) {
      using U = uint64_t;
      U v = __ldg(reinterpret_cast<const U*>(ptr));
      return *reinterpret_cast<const T*>(&v);
    }
  }

}

#endif /* LIBINTX_CUDA_API_INTRINSICS_H */
