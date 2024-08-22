#include "libintx/gpu/api/api.h"
#include "libintx/gpu/api/runtime.h"

#include <cstring>

int libintx::gpu::current_device::get() {
  int device;
  LIBINTX_GPU_API(GetDevice, &device);
  return device;
}

void libintx::gpu::current_device::set(int device) {
  LIBINTX_GPU_API(SetDevice, device);
}

int libintx::gpu::device::count() {
  int count = -1;
  LIBINTX_GPU_API(GetDeviceCount, &count);
  return count;
}

void libintx::gpu::stream::synchronize(gpuStream_t stream) {
  LIBINTX_GPU_API(StreamSynchronize, stream);
}

void* libintx::gpu::device_memory_t::allocate(size_t bytes) {
  void *ptr = nullptr;
  LIBINTX_GPU_API(Malloc, &ptr, bytes);
  return ptr;
}

void libintx::gpu::device_memory_t::free(void *ptr) {
  LIBINTX_GPU_API(Free,ptr);
}

void libintx::gpu::memcpy(void *dst, const void *src, size_t bytes) {
  LIBINTX_GPU_API(Memcpy, dst, src, bytes, gpuMemcpyDefault);
}

void libintx::gpu::memcpy(void *dst, const void *src, size_t bytes, gpuStream_t stream) {
  LIBINTX_GPU_API(MemcpyAsync, dst, src, bytes, gpuMemcpyDefault, stream);
}

void* libintx::gpu::host_memory_t::allocate(size_t bytes) {
  void *ptr = nullptr;
  LIBINTX_GPU_API(MallocHost, &ptr, bytes);
  return ptr;
}

void libintx::gpu::host_memory_t::free(void *ptr) {
  LIBINTX_GPU_API(FreeHost, ptr);
}

void libintx::gpu::host_memory_t::memset(void *dst, const int value, size_t bytes) {
  std::memset(dst, value, bytes);
}

void libintx::gpu::device_memory_t::memset(void *dst, const int value, size_t bytes) {
  LIBINTX_GPU_API(Memset, dst, value, bytes);
}

void libintx::gpu::device_memory_t::memset(void *dst, const int value, size_t bytes, gpuStream_t stream) {
  LIBINTX_GPU_API(MemsetAsync, dst, value, bytes, stream);
}

template<>
void* libintx::gpu::host::device_pointer(void* ptr) {
  void *device_ptr = nullptr;
  LIBINTX_GPU_API(HostGetDevicePointer, &device_ptr, ptr, 0);
  return device_ptr;
}

template<>
void libintx::gpu::host::register_pointer(const void *ptr, size_t size) {
  LIBINTX_GPU_API(HostRegister, const_cast<void*>(ptr), size, gpuHostRegisterDefault);
}

void libintx::gpu::host::unregister_pointer(const void *ptr) {
  LIBINTX_GPU_API(HostUnregister, const_cast<void*>(ptr));
}
