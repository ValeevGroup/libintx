math(EXPR lmax2 "2*${LIBINTX_MAX_L}")

# foreach (KET RANGE ${lmax2})

#   foreach (BRA RANGE ${lmax2})
#     set(target libintx.md4.kernel.${BRA}_${KET})
#     add_library(${target} OBJECT EXCLUDE_FROM_ALL md4.kernel.cc)
#     target_compile_definitions(
#       ${target}
#       PRIVATE
#       LIBINTX_MD_MD4_KERNEL_BRA_KET=${BRA}\,${KET}
#       )
#     list(APPEND libintx_md_md4_sources $<TARGET_OBJECTS:${target}>)
#   endforeach()

#   foreach (X RANGE ${LIBINTX_MAX_X})
#     set(target libintx.md3.kernel.${X}_${KET})
#     add_library(${target} OBJECT EXCLUDE_FROM_ALL md3.kernel.cc)
#     target_compile_definitions(
#       ${target}
#       PRIVATE
#       LIBINTX_MD_MD3_KERNEL_BRA_KET=${X}\,${KET}
#       )
#     list(APPEND libintx_md_md3_sources $<TARGET_OBJECTS:${target}>)
#   endforeach()

# endforeach()

foreach (Centers 2)
  add_library(
    libintx.md${Centers}
    EXCLUDE_FROM_ALL
    md${Centers}.cc
    ${libintx_md_md${Centers}_sources}
    #basis.cc
    )
  set_target_properties(libintx.md${Centers} PROPERTIES OUTPUT_NAME intx.md${Centers})
  target_link_libraries(libintx.md${Centers} ${LIBINTX_OPENMP_LIBRARIES})
  install(TARGETS libintx.md${Centers})
endforeach()

# target_link_libraries(libintx.md3 libintx.blas)
# target_link_libraries(libintx.md4 libintx.blas)
# target_link_libraries(libintx.md3 libintx.md.basis)
# target_link_libraries(libintx.md4 libintx.md.basis)

# install(TARGETS libintx.md.basis)
# install(TARGETS libintx.md.eri3)
# install(TARGETS libintx.md.md4)
#install(FILES engine.h DESTINATION libintx/cuda/md)
