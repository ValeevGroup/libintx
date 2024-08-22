if (LIBINTX_HIP)
  set_source_files_properties(
    basis.cu
    md3.kernel.cu
    md4.kernel.cu
    PROPERTIES
    LANGUAGE HIP
    )
endif()

math(EXPR lmax2 "2*${LIBINTX_MAX_L}")

foreach (KET RANGE ${lmax2})

  foreach (BRA RANGE ${lmax2})
    set(target libintx.gpu.md.eri4.kernel.${BRA}_${KET})
    add_library(${target} OBJECT EXCLUDE_FROM_ALL md4.kernel.cu)
    target_compile_definitions(
      ${target}
      PRIVATE
      LIBINTX_GPU_MD_MD4_KERNEL_BRA=${BRA}
      LIBINTX_GPU_MD_MD4_KERNEL_KET=${KET}
      LIBINTX_GPU_MAX_SHMEM=${LIBINTX_GPU_MAX_SHMEM}
      )
    list(APPEND libintx.gpu_md_eri4_sources $<TARGET_OBJECTS:${target}>)
  endforeach()

  foreach (X RANGE ${LIBINTX_MAX_X})
    set(target libintx.gpu.md.eri3.kernel.${X}_${KET})
    add_library(${target} OBJECT EXCLUDE_FROM_ALL md3.kernel.cu)
    target_compile_definitions(
      ${target}
      PRIVATE
      LIBINTX_GPU_MD_MD3_KERNEL_X=${X}
      LIBINTX_GPU_MD_MD3_KERNEL_KET=${KET}
      LIBINTX_GPU_MAX_SHMEM=${LIBINTX_GPU_MAX_SHMEM}
      )
    list(APPEND libintx.gpu_md_eri3_sources $<TARGET_OBJECTS:${target}>)
  endforeach()

endforeach()

add_library(
  libintx.gpu.md.basis
  EXCLUDE_FROM_ALL
  basis.cu
  )
set_target_properties(libintx.gpu.md.basis PROPERTIES OUTPUT_NAME intx.gpu.md.basis)

add_library(
  libintx.gpu.md.eri3
  EXCLUDE_FROM_ALL
  md3.cc
  ${libintx.gpu_md_eri3_sources}
  )
set_target_properties(libintx.gpu.md.eri3 PROPERTIES OUTPUT_NAME intx.gpu.md.eri3)
target_link_libraries(libintx.gpu.md.eri3 libintx.gpu)
target_link_libraries(libintx.gpu.md.eri3 libintx.gpu.md.basis)
#target_link_libraries(libintx.gpu.md CUDA::cublas)

add_library(
  libintx.gpu.md.eri4
  EXCLUDE_FROM_ALL
  md4.cc
  ${libintx.gpu_md_eri4_sources}
  )
set_target_properties(libintx.gpu.md.eri4 PROPERTIES OUTPUT_NAME intx.gpu.md.eri4)
target_link_libraries(libintx.gpu.md.eri4 libintx.gpu)
target_link_libraries(libintx.gpu.md.eri4 libintx.gpu.md.basis)
#target_link_libraries(libintx.gpu.md CUDA::cublas)

install(TARGETS libintx.gpu.md.basis)
install(TARGETS libintx.gpu.md.eri3)
install(TARGETS libintx.gpu.md.eri4)
install(FILES engine.h DESTINATION libintx.gpu/md)
