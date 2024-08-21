#ifndef LIBINTX_GPU_API_RUNTIME_H
#define LIBINTX_GPU_API_RUNTIME_H

#include "libintx/gpu/api/forward.h"

#include <memory>
#include <cstdio>
#include <cuda_runtime.h>

#define LIBINTX_GPU_API(SYMBOL, ...)                                    \
  if (auto err = cuda ## SYMBOL (__VA_ARGS__); err != cudaSuccess) {    \
    char msg[128];                                                      \
    snprintf(                                                           \
      msg, sizeof(msg),                                                 \
      "cuda" #SYMBOL " returned with %s: %s",                           \
      cudaGetErrorName(err), cudaGetErrorString(err)                    \
    );                                                                  \
    throw libintx::gpu::runtime_error(msg);                             \
  }

namespace libintx::gpu {

  struct Stream {

    Stream(const Stream&) = delete;
    Stream(Stream&&) = delete;
    Stream& operator=(const Stream&) = delete;
    Stream& operator=(Stream&&) = delete;

    Stream() {
      LIBINTX_GPU_API(StreamCreate, &this->stream_);
    }

    ~Stream() throw() {
      try {
        LIBINTX_GPU_API(StreamDestroy, this->stream_);
      }
      catch (...) {}
    }

    operator gpuStream_t() const { return this->stream_; }

    void synchronize() {
      LIBINTX_GPU_API(StreamSynchronize, this->stream_);
    }

    void wait(gpuEvent_t event) {
      LIBINTX_GPU_API(StreamWaitEvent, this->stream_, event);
    }

    template<typename F>
    void add_callback(F &&callback) {
      constexpr auto callback_adapter = [](gpuStream_t stream, cudaError_t error, void *data) {
        auto callback = ::std::unique_ptr<F>(reinterpret_cast<F*>(data));
        (*callback)(stream, error);
      };
      auto f = std::make_unique<F>(callback);
      LIBINTX_GPU_API(StreamAddCallback, this->stream_, callback_adapter, f.release(), 0);
    }

  private:
    gpuStream_t stream_;

  };

  struct Event {

    Event(const Event&) = delete;
    Event(Event&&) = delete;
    Event& operator=(const Event&) = delete;
    Event& operator=(Event&&) = delete;

    explicit Event() {
      LIBINTX_GPU_API(EventCreate, &this->event_);
    }

    ~Event() {
      try {
        LIBINTX_GPU_API(EventDestroy, this->event_);
      }
      catch (...) {}
    }

    void synchronize() {
      LIBINTX_GPU_API(EventSynchronize, this->event_);
    }

    operator gpuEvent_t() const { return this->event_; }

  private:
    gpuEvent_t event_;
  };

}

#endif /* LIBINTX_GPU_API_RUNTIME_H */
