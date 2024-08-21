#ifndef LIBINTX_GPU_API_THREAD_GROUP_H
#define LIBINTX_GPU_API_THREAD_GROUP_H

#include <cooperative_groups.h>

namespace libintx::gpu {

  using cooperative_groups::thread_group;
  using cooperative_groups::tiled_partition;
  using cooperative_groups::this_thread_block;

  template<int X, int Y = 1, int Z = 1>
  struct thread_block {
    static constexpr int x = X;
    static constexpr int y = Y;
    static constexpr int z = Z;
    __device__ static constexpr int size() { return x*y*z; }
    __device__ static auto thread_rank() {
      return this_thread_block().thread_rank();
    }
    __device__ static void sync() {
      this_thread_block().sync();
    }
    constexpr operator dim3() const {
      return dim3{ x, y, z };
    }
    __device__ operator thread_group() const {
      return this_thread_block();
    }
  };

  struct this_warp {
    __device__ static constexpr int size() { return 32; }
    __device__ static auto thread_rank() {
      return this_thread_block().thread_rank()%32;
    }
    __device__ static void sync() {
      __syncwarp();
    }
  };

  template<int DimX, int ... Dims>
  __device__ __forceinline__
  constexpr auto this_thread_block() {
    return thread_block<DimX,Dims...>{};
  }

}

namespace libintx::gpu {

  template<typename T, typename V = T>
  __device__
  void fill(size_t size, T *dst, V v, const auto &thread_group) {
    for (int i = thread_group.thread_rank(); i < size; i += thread_group.size()) {
      dst[i] = v;
    }
  }

  template<typename T>
  __device__ __forceinline__
  void memcpy(size_t size, const T *src, T *dst, const auto &thread_group) {
    for (int i = thread_group.thread_rank(); i < size; i += thread_group.size()) {
      dst[i] = src[i];
    }
  }

  template<typename T>
  __device__ __forceinline__
  void memcpy1(const T *src, T *dst, const auto &thread_group) {
    using byte4 = uint32_t;
    static_assert(sizeof(T)%sizeof(byte4) == 0);
    memcpy(
      sizeof(T)/sizeof(byte4),
      reinterpret_cast<const byte4*>(src),
      reinterpret_cast<byte4*>(dst),
      thread_group
    );
  }

  template<typename T, typename V>
  __device__ __forceinline__
  void memset1(T *dst, V value, const auto &thread_group) {
    static_assert(sizeof(T)%sizeof(V) == 0);
    fill(
      sizeof(T)/sizeof(V),
      reinterpret_cast<V*>(dst),
      value,
      thread_group
    );
  }

}

#endif /* LIBINTX_GPU_API_THREAD_GROUP_H */
