#ifndef LIBINTX_CUDA_STREAM_H
#define LIBINTX_CUDA_STREAM_H

#include <cuda/api/stream.hpp>
#include <vector>
#include <tuple>

namespace libintx::gpu {

  struct Stream : ::cuda::stream_t<> {
    Stream(int device = 0)
      : ::cuda::stream_t<>(create(device)) {}
    operator cudaStream_t() const {
      return this->id();
    }
    template<class F>
    void add_callback(F f) { this->enqueue.callback(f); }
  private:
    static ::cuda::stream_t<> create(int device) {
      return ::cuda::stream::create(device, false);
    }
  };

  template<typename ... Ts>
  struct StreamPool {
    StreamPool(size_t size = 1) : pool_(size) {}
    auto begin() { return pool_.begin(); }
    auto end() { return pool_.end(); }
    auto& next() {
      if (next_ >= pool_.size()) next_ = 0;
      return pool_.at(next_++);
    }
    void synchronize() {
      for (auto &t : pool_) {
        std::get<0>(t).synchronize();
      }
    }
  private:
    std::vector< std::tuple<Stream,Ts...> > pool_;
    size_t next_ = 0;
  };

}

namespace libintx::gpu::stream {

  inline void synchronize() {
    auto status = ::cudaStreamSynchronize(0);
    ::cuda::throw_if_error(
      status,
      ::std::string("Failed synchronizing default stream")
    );
  }

}

#endif /* LIBINTX_CUDA_STREAM_H */
