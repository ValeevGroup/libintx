#ifndef LIBINTX_CUDA_API_THREAD_GROUP_H
#define LIBINTX_CUDA_API_THREAD_GROUP_H

#include <cooperative_groups.h>

namespace libintx::cuda {
  using cooperative_groups::thread_block;
  using cooperative_groups::thread_group;
  using cooperative_groups::this_thread_block;
  using cooperative_groups::coalesced_threads;
}

namespace libintx::cuda {

  template<typename T, typename V = T>
  __device__
  void fill(size_t size, T *dst, V v, const thread_group &g) {
    for (int i = g.thread_rank(); i < size; i += g.size()) {
      dst[i] = v;
    }
  }

  template<typename T>
  __device__
  void memcpy(size_t size, const T *src, T *dst, const thread_group &block) {
    for (int i = block.thread_rank(); i < size; i += block.size()) {
      dst[i] = src[i];
    }
  }

  template<typename T>
  __device__
  void memcpy1(const T *src, T *dst, const thread_group &block) {
    using byte4 = uint32_t;
    static_assert(sizeof(T)%sizeof(byte4) == 0);
    memcpy(
      sizeof(T)/sizeof(byte4),
      reinterpret_cast<const byte4*>(src),
      reinterpret_cast<byte4*>(dst),
      block
    );
  }


}

#endif /* LIBINTX_CUDA_API_THREAD_GROUP_H */
