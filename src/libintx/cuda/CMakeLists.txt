enable_language(CUDA)
find_package(CUDAToolkit 11 REQUIRED)

set(maxrregcount 128)
if (maxrregcount)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=${maxrregcount}")
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(
  libintx.cuda
  api/api.cc
  boys.cc
  blas.cu
  )
set_target_properties(libintx.cuda PROPERTIES OUTPUT_NAME intx.cuda)
target_link_libraries(libintx.cuda libintx.boys)
target_link_libraries(libintx.cuda CUDA::cudart)

math(EXPR lmax2 "2*${LIBINTX_MAX_L}")
foreach (L RANGE ${lmax2})
  foreach (X RANGE ${LIBINTX_MAX_X})
    set(target libintx.cuda.jengine.kernel.${L}_${X})
    add_library(${target} OBJECT EXCLUDE_FROM_ALL jengine/md/jengine.kernel.cu)
    target_compile_definitions(
      ${target}
      PRIVATE
      LIBINTX_CUDA_MD_JENGINE_KERNEL_BRA_KET=${L}\\,${X}
      )
    list(APPEND libintx_cuda_jengine_sources $<TARGET_OBJECTS:${target}>)
  endforeach()
endforeach()

add_library(
  libintx.cuda.jengine
  EXCLUDE_FROM_ALL
  jengine/md/jengine.cpp
  jengine/md/hermite.cu
  ${libintx_cuda_jengine_sources}
  )
set_target_properties(libintx.cuda.jengine PROPERTIES OUTPUT_NAME intx.cuda.jengine)
target_link_libraries(libintx.cuda.jengine libintx.cuda)

install(TARGETS libintx.cuda)
install(FILES forward.h DESTINATION libintx/cuda)

add_subdirectory(md)
