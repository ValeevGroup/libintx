add_library(libintx.os engine.cc)
target_link_libraries(libintx.os libintx.boys)
set_target_properties(libintx.os PROPERTIES OUTPUT_NAME intx.os)
