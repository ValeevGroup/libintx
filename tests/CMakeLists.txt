add_custom_target(all.tests)

include_directories(${PROJECT_SOURCE_DIR}/include/eigen3)
include_directories(${PROJECT_SOURCE_DIR}/tests)

macro(add_libintx_test name source)
  add_executable(${name} EXCLUDE_FROM_ALL ${source})
  add_test(NAME ${name} COMMAND ${name})
  add_dependencies(all.tests ${name})
endmacro()

add_libintx_test(interpolate.test interpolate.test.cc)
target_link_libraries(interpolate.test libboys)

add_libintx_test(boys.test boys.test.cc)
target_link_libraries(boys.test libboys)

add_libintx_test(libintx.md.test libintx.md.test.cc)
target_link_libraries(libintx.md.test libboys)

option(LIBINTX_LIBINT2 "" OFF)
if (LIBINTX_LIBINT2)
  find_package(Libint2 REQUIRED)
  add_library(libintx.test.reference libint2.reference.cc)
  set_target_properties(libintx.test.reference PROPERTIES OUTPUT_NAME intx.test.reference)
  target_compile_definitions(libintx.test.reference PUBLIC LIBINTX_TEST_REFERENCE)
  target_link_libraries(libintx.test.reference PRIVATE Libint2::cxx)
endif()

if (LIBINTX_CUDA)
  enable_language(CUDA)
  add_libintx_test(boys.gpu.test boys.gpu.test.cu)
  target_link_libraries(boys.gpu.test libboys libintx.gpu)
endif()

foreach (n 3 4)
  if (NOT TARGET libintx.gpu.md${n})
    continue()
  endif()
  # tests
  add_libintx_test(libintx.gpu.md${n}.test libintx.gpu.md${n}.test.cc)
  target_link_libraries(libintx.gpu.md${n}.test libintx.gpu.md${n})
  # perf benchmarks
  add_executable(libintx.gpu.md${n}.perf EXCLUDE_FROM_ALL libintx.gpu.md${n}.perf.cc)
  target_link_libraries(libintx.gpu.md${n}.perf libintx.gpu.md${n})
  if (TARGET libintx.test.reference)
    target_link_libraries(libintx.gpu.md${n}.perf libintx.test.reference)
  endif()
endforeach()

if (TARGET libintx.gpu.jengine)
  add_libintx_test(libintx.gpu.jengine.test libintx.gpu.jengine.test.cc)
  target_link_libraries(libintx.gpu.jengine.test libintx.gpu.jengine)
endif()
