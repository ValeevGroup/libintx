add_library(
  libintx.ao.screening
  screening.cc
  )
set_target_properties(libintx.ao.screening PROPERTIES OUTPUT_NAME intx.ao.screening)
target_link_libraries(libintx.ao.screening ${LIBINTX_OPENMP_LIBRARIES})

add_subdirectory(md)
