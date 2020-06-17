find_package(Libint2 REQUIRED)

add_library(libintx.libint2 EXCLUDE_FROM_ALL engine.cc)
target_link_libraries(libintx.libint2 PUBLIC Libint2::cxx)
set_target_properties(libintx.libint2 PROPERTIES OUTPUT_NAME intx.libint2)
