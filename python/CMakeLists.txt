add_subdirectory(pybind11)

pybind11_add_module(
  boys-python EXCLUDE_FROM_ALL
  src/boys/boys.cc
)
set_target_properties(boys-python PROPERTIES OUTPUT_NAME boys)
target_link_libraries(boys-python PRIVATE libintx.boys)

pybind11_add_module(
  libintx-python EXCLUDE_FROM_ALL
  src/libintx/libintx.cc
)
set_target_properties(libintx-python PROPERTIES OUTPUT_NAME libintx)

if (LIBINTX_GPU)
  target_link_libraries(libintx-python PRIVATE libintx.gpu.md.eri3)
  target_link_libraries(libintx-python PRIVATE libintx.gpu.md.eri4)
endif()
