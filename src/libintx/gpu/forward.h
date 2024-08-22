#ifndef LIBINTX_GPU_FORWARD_H
#define LIBINTX_GPU_FORWARD_H

#include "libintx/config.h"

#ifdef LIBINTX_CUDA

struct CUstream_st;
namespace libintx { typedef CUstream_st* gpuStream_t; }

#endif

#ifdef LIBINTX_HIP

struct ihipStream_t;
namespace libintx { typedef ihipStream_t* gpuStream_t; }

#endif

#endif /* LIBINTX_GPU_FORWARD_H */
