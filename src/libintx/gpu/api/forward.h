#ifndef LIBINTX_GPU_API_FORWARD_H
#define LIBINTX_GPU_API_FORWARD_H

#include "libintx/gpu/api/config.h"

#ifdef LIBINTX_GPU_API_CUDA

struct CUstream_st;
struct CUevent_st;

namespace libintx {

  typedef CUstream_st* gpuStream_t;
  typedef CUevent_st* gpuEvent_t;

}

#endif

#ifdef LIBINTX_GPU_API_HIP

struct ihipStream_t;
struct ihipEvent_t;

namespace libintx {

  typedef ihipStream_t* gpuStream_t;
  typedef ihipEvent_t* gpuEvent_t;

}

#endif

#endif /* LIBINTX_GPU_API_FORWARD_H */
