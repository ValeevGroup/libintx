enable_language(CUDA)
find_package(CUDAToolkit)

math(EXPR lmax2 "2*${LIBINTX_MAX_L}")
foreach (BRA RANGE ${lmax2})
  foreach (KET RANGE ${lmax2})
    set(target libintx.cuda.md.eri4.kernel.${BRA}_${KET})
    add_library(${target} OBJECT EXCLUDE_FROM_ALL md4.kernel.cu)
    target_compile_definitions(
      ${target}
      PRIVATE
      LIBINTX_CUDA_MD_MD4_KERNEL_BRA_KET=${BRA}\\,${KET}
      )
    list(APPEND libintx_cuda_md_eri4_sources $<TARGET_OBJECTS:${target}>)
  endforeach()
endforeach()

add_library(
  libintx.cuda.md.basis
  EXCLUDE_FROM_ALL
  basis.cu
  )

add_library(
  libintx.cuda.md.eri4
  EXCLUDE_FROM_ALL
  md4.cc
  ${libintx_cuda_md_eri4_sources}
  )

set_target_properties(libintx.cuda.md.eri4 PROPERTIES OUTPUT_NAME intx.cuda.md.eri4)
target_link_libraries(libintx.cuda.md.eri4 libintx.cuda)
target_link_libraries(libintx.cuda.md.eri4 libintx.cuda.md.basis)
#target_link_libraries(libintx.cuda.md CUDA::cublas)
