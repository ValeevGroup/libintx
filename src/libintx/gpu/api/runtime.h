#ifndef LIBINTX_GPU_API_RUNTIME_H
#define LIBINTX_GPU_API_RUNTIME_H

#include "libintx/gpu/api/forward.h"
#include "libintx/gpu/api/api.h"

#include <memory>
#include <cstdio>

#ifdef LIBINTX_GPU_API_CUDA
#include <cuda_runtime.h>
#define LIBINTX_GPU_API_NAME "cuda"
#define LIBINTX_GPU_API_SYMBOL(SYMBOL) cuda ## SYMBOL
#endif

#ifdef LIBINTX_GPU_API_HIP
#include <hip/hip_runtime.h>
#define LIBINTX_GPU_API_NAME "hip"
#define LIBINTX_GPU_API_SYMBOL(SYMBOL) hip ## SYMBOL
#endif

using gpuError_t = LIBINTX_GPU_API_SYMBOL(Error_t);
constexpr auto gpuSuccess = LIBINTX_GPU_API_SYMBOL(Success);
constexpr auto gpuMemcpyDefault = LIBINTX_GPU_API_SYMBOL(MemcpyDefault);
constexpr auto gpuHostRegisterDefault = LIBINTX_GPU_API_SYMBOL(HostRegisterDefault);

#define LIBINTX_GPU_API(SYMBOL, ...)                                            \
  if (auto err = LIBINTX_GPU_API_SYMBOL(SYMBOL) (__VA_ARGS__); err != gpuSuccess) { \
  char msg[128];                                                                \
  snprintf(                                                                     \
    msg, sizeof(msg),                                                           \
    "%s%s returned with %s: %s",                                                \
    LIBINTX_GPU_API_NAME, #SYMBOL,                                              \
    LIBINTX_GPU_API_SYMBOL(GetErrorName)(err),                                  \
    LIBINTX_GPU_API_SYMBOL(GetErrorString)(err)                                 \
  );                                                                            \
  throw libintx::gpu::runtime_error(msg);                                       \
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
      constexpr auto callback_adapter = [](gpuStream_t stream, gpuError_t error, void *data) {
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
