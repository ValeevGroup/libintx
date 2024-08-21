#include "libintx/gpu/api/api.h"
#include "libintx/gpu/api/stream.h"
#include <cuda/api_wrappers.hpp>

void libintx::gpu::memcpy(void *dst, const void *src, size_t bytes) {
  ::cuda::memory::copy(dst, src, bytes);
}

void libintx::gpu::memcpy(void *dst, const void *src, size_t bytes, Stream& stream) {
  ::cuda::memory::async::copy(dst, src, bytes, stream);
}

void* libintx::gpu::host_memory_t::allocate(size_t bytes) {
  return ::cuda::memory::host::make_unique<char[]>(bytes).release();
}

void libintx::gpu::host_memory_t::free(void *ptr) {
  ::cuda::memory::host::free(ptr);
}

void libintx::gpu::host_memory_t::memset(void *dst, const int value, size_t bytes) {
  std::memset(dst, value, bytes);
}


void* libintx::gpu::device_memory_t::allocate(size_t bytes) {
  auto current_device = ::cuda::device::current::get();
  return ::cuda::memory::device::make_unique<char[]>(current_device, bytes).release();
}

void libintx::gpu::device_memory_t::free(void *ptr) {
  ::cuda::memory::device::free(ptr);
}

void libintx::gpu::device_memory_t::memset(void *dst, const int value, size_t bytes) {
  ::cuda::memory::device::set(dst, value, bytes);
}

void libintx::gpu::device_memory_t::memset(void *dst, const int value, size_t bytes, Stream& stream) {
  ::cuda::memory::device::async::set(dst, value, bytes, stream);
}

void libintx::gpu::stream::synchronize(cudaStream_t stream) {
  auto status = ::cudaStreamSynchronize(stream);
  ::cuda::throw_if_error(
    status,
    ::std::string("Failed synchronizing stream")
  );
  //return (::cuda::outstanding_error::get() == ::cuda::status::success);
}

bool libintx::gpu::device::synchronize() {
  ::cuda::device::current::get().synchronize();
  return (::cuda::outstanding_error::get() == ::cuda::status::success);
}

void libintx::gpu::error::ensure_none(const char *s) {
  ::cuda::outstanding_error::ensure_none(s);
}

template<>
void* libintx::gpu::host::device_pointer(void* ptr) {
  void *device_ptr;
  cudaHostGetDevicePointer(&device_ptr, ptr, 0);
  ::cuda::outstanding_error::ensure_none();
  return device_ptr;
}

template<>
void libintx::gpu::host::register_pointer(const void *ptr, size_t size) {
  cudaHostRegister(const_cast<void*>(ptr), size, cudaHostRegisterDefault);
  ::cuda::outstanding_error::ensure_none();
}

void libintx::gpu::host::unregister_pointer(const void *ptr) {
  cudaHostUnregister(const_cast<void*>(ptr));
  ::cuda::outstanding_error::ensure_none();
}


void libintx::gpu::kernel::set_prefered_shared_memory_carveout(const void *f, size_t carveout) {
  cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
  ::cuda::outstanding_error::ensure_none();
}

void libintx::gpu::kernel::set_max_dynamic_shared_memory_size(const void *f, size_t bytes) {
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  ::cuda::outstanding_error::ensure_none();
}
