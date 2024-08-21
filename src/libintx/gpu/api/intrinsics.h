#ifndef LIBINTX_GPU_API_INTRINSICS_H
#define LIBINTX_GPU_API_INTRINSICS_H

namespace libintx::gpu {

  template<typename T>
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

#endif /* LIBINTX_GPU_API_INTRINSICS_H */
