add_library(libintx.rysq EXCLUDE_FROM_ALL engine.cc)
set_target_properties(libintx.rysq PROPERTIES OUTPUT_NAME intx.rysq)
target_include_directories(libintx.rysq PRIVATE ..)
