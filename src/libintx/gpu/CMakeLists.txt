enable_language(CUDA)
find_package(CUDAToolkit 11 REQUIRED)

set(maxrregcount 128)
if (maxrregcount)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=${maxrregcount}")
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(
  libintx.gpu
  api/api.cc
  boys.cc
  blas.cu
  )
set_target_properties(libintx.gpu PROPERTIES OUTPUT_NAME intx.gpu)
target_link_libraries(libintx.gpu libintx.boys)
target_link_libraries(libintx.gpu CUDA::cudart)

math(EXPR lmax2 "2*${LIBINTX_MAX_L}")
foreach (L RANGE ${lmax2})
  foreach (X RANGE ${LIBINTX_MAX_X})
    set(target libintx.gpu.jengine.kernel.${L}_${X})
    add_library(${target} OBJECT EXCLUDE_FROM_ALL jengine/md/jengine.kernel.cu)
    target_compile_definitions(
      ${target}
      PRIVATE
      LIBINTX_CUDA_MD_JENGINE_KERNEL_BRA_KET=${L}\\,${X}
      )
    list(APPEND libintx_gpu_jengine_sources $<TARGET_OBJECTS:${target}>)
  endforeach()
endforeach()

add_library(
  libintx.gpu.jengine
  EXCLUDE_FROM_ALL
  jengine/md/jengine.cpp
  jengine/md/hermite.cu
  ${libintx_gpu_jengine_sources}
  )
set_target_properties(libintx.gpu.jengine PROPERTIES OUTPUT_NAME intx.gpu.jengine)
target_link_libraries(libintx.gpu.jengine libintx.gpu)

install(TARGETS libintx.gpu)
install(FILES forward.h DESTINATION libintx/gpu)

add_subdirectory(md)
