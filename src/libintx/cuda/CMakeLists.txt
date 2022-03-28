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
  )
set_target_properties(libintx.cuda PROPERTIES OUTPUT_NAME intx.cuda)
target_link_libraries(libintx.cuda libintx.boys CUDA::cudart)

add_library(
  libintx.cuda.eri
  EXCLUDE_FROM_ALL
  eri/eri1.cu
  eri/eri2.cu
  eri/eri3.cu
  eri/eri4.cu
  )
set_target_properties(libintx.cuda.eri PROPERTIES OUTPUT_NAME intx.cuda.eri)
target_link_libraries(libintx.cuda.eri libintx.cuda)

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
  jengine/md/hermitian.cu
  ${libintx_cuda_jengine_sources}
  )
set_target_properties(libintx.cuda.jengine PROPERTIES OUTPUT_NAME intx.cuda.jengine)
target_link_libraries(libintx.cuda.jengine libintx.cuda)
