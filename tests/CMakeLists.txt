add_executable(interpolate.test EXCLUDE_FROM_ALL interpolate.test.cc)
target_link_libraries(interpolate.test libboys)
add_test(NAME interpolate COMMAND interpolate.test)

add_executable(boys.test EXCLUDE_FROM_ALL boys.test.cc)
target_link_libraries(boys.test libboys)
add_test(NAME boys COMMAND boys.test)

if (LIBINTX_ENABLE_CUDA)
  add_executable(boys.cuda.test EXCLUDE_FROM_ALL boys.cuda.test.cu)
  target_link_libraries(boys.cuda.test libboys)
  add_test(NAME boys.cuda COMMAND boys.cuda.test)
endif()

add_executable(libintx.benchmarks EXCLUDE_FROM_ALL libintx.benchmarks.cc)
target_link_libraries(libintx.benchmarks libintx.os)
if (TARGET libintx.libint2)
  add_definitions(-DLIBINTX_LIBINT2)
  set(libintx_test_reference libintx.libint2)
  target_link_libraries(libintx.benchmarks libintx.libint2)
endif()

if (NOT libintx_test_reference)
  set(libintx_test_reference libintx.rysq)
endif()

add_executable(libintx.test EXCLUDE_FROM_ALL libintx.test.cc)
target_link_libraries(libintx.test libintx.os ${libintx_test_reference})
add_test(NAME libintx COMMAND libintx.test)

if (TARGET libintx.cuda)
  add_executable(libintx.cuda.test EXCLUDE_FROM_ALL libintx.cuda.test.cc)
  target_link_libraries(libintx.cuda.test libintx.cuda ${libintx_test_reference})
  add_test(NAME libintx.cuda COMMAND libintx.cuda.test)
  add_executable(libintx.cuda.benchmarks EXCLUDE_FROM_ALL libintx.cuda.benchmarks.cc)
  target_link_libraries(libintx.cuda.benchmarks libintx.cuda libintx.os)
endif()

