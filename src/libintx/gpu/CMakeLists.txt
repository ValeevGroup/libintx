if (LIBINTX_CUDA)

  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CMAKE_CUDA_ARCHITECTURES")
  endif()

  set(CMAKE_CUDA_STANDARD 20)
  if (CMAKE_VERSION VERSION_LESS 3.25.2)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")
  else()
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -expt-relaxed-constexpr --resource-usage")
  add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Release>>>:-src-in-ptx;-lineinfo>")

  enable_language(CUDA)
  find_package(CUDAToolkit 12 REQUIRED)

  #include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  set(LIBINTX_GPU_API_CUDA ON)
  set(libintx_gpu_runtime CUDA::cudart)

endif(LIBINTX_CUDA)

if (LIBINTX_HIP)

  enable_language(HIP)
  find_package(hip REQUIRED)
  #message("*** " ${hip_FOUND})
  set(LIBINTX_GPU_API_HIP ON)
  set(libintx_gpu_runtime hip::host)

  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage")

  set_source_files_properties(
    blas.cu
    jengine/md/jengine.kernel.cu
    jengine/md/hermite.cu
    PROPERTIES
    LANGUAGE HIP
    )

endif(LIBINTX_HIP)

if (NOT DEFINED LIBINTX_GPU_MAX_SHMEM)
  set (max_shmem 49152)
  if (LIBINTX_HIP)
    set (max_shmem 32768)
  endif()
  set(LIBINTX_GPU_MAX_SHMEM ${max_shmem} CACHE STRING "Maximum GPU shared memory/LDS per thread-block")
endif()

configure_file(api/config.h.in ${PROJECT_BINARY_DIR}/libintx/gpu/api/config.h)

add_library(
  libintx.gpu
  api/api.cc
  boys.cc
  blas.cu
  )
set_target_properties(libintx.gpu PROPERTIES OUTPUT_NAME intx.gpu)
target_link_libraries(libintx.gpu PRIVATE libintx.boys)
target_link_libraries(libintx.gpu PRIVATE ${libintx_gpu_runtime})

math(EXPR lmax2 "2*${LIBINTX_MAX_L}")
foreach (BRA RANGE ${lmax2})
  foreach (X RANGE ${LIBINTX_MAX_X})
    set(target libintx.gpu.jengine.kernel.${BRA}_${X})
    add_library(${target} OBJECT EXCLUDE_FROM_ALL jengine/md/jengine.kernel.cu)
    target_compile_definitions(
      ${target}
      PRIVATE
      LIBINTX_GPU_MD_JENGINE_KERNEL_BRA=${BRA}
      LIBINTX_GPU_MD_JENGINE_KERNEL_KET=${X}
      )
    list(APPEND libintx_gpu_jengine_sources $<TARGET_OBJECTS:${target}>)
  endforeach()
endforeach()

add_library(
  libintx.gpu.jengine
  EXCLUDE_FROM_ALL
  jengine/md/jengine.cc
  jengine/md/hermite.cu
  ${libintx_gpu_jengine_sources}
  )
set_target_properties(libintx.gpu.jengine PROPERTIES OUTPUT_NAME intx.gpu.jengine)
target_link_libraries(libintx.gpu.jengine PRIVATE ${libintx_gpu_runtime})
target_link_libraries(libintx.gpu.jengine PRIVATE libintx.gpu)

install(TARGETS libintx.gpu)
install(FILES forward.h DESTINATION libintx/gpu)

add_subdirectory(md)
