add_library(libintx.blas blas.cc)

if (LIBINTX_CBLAS_H)
  target_compile_definitions(libintx.blas PRIVATE LIBINTX_CBLAS_H=${LIBINTX_CBLAS_H})
endif()

if (LIBINTX_LAPACKE_H)
  target_compile_definitions(libintx.blas PRIVATE LIBINTX_LAPACKE_H=${LIBINTX_LAPACKE_H})
endif()

if (LIBINTX_INTEL_MKL)
  target_compile_definitions(libintx.blas PRIVATE LIBINTX_INTEL_MKL)
  target_link_libraries(libintx.blas mkl_rt)
endif()

if (LIBINTX_APPLE_ACCELERATE)
  target_compile_definitions(libintx.blas PRIVATE LIBINTX_APPLE_ACCELERATE)
endif()

if (TARGET LAPACK::LAPACK)
  target_link_libraries(libintx.blas LAPACK::LAPACK)
endif()

install(
  FILES forward.h shell.h orbital.h array.h math.h engine.h utility.h
  DESTINATION libintx
  )
