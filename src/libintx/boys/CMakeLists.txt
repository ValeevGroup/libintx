add_library(libintx.boys boys.cc)
set_target_properties(libintx.boys PROPERTIES OUTPUT_NAME intx.boys)

add_library(libboys EXCLUDE_FROM_ALL boys.cc)
set_target_properties(libboys PROPERTIES OUTPUT_NAME boys)

