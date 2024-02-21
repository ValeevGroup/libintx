include_directories(${PROJECT_SOURCE_DIR}/include/eigen3)

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

if (TARGET libintx.cuda.md.eri4)
  # benchmarks
  add_executable(libintx.cuda.md4.benchmarks EXCLUDE_FROM_ALL libintx.cuda.md4.benchmarks.cc)
  target_link_libraries(libintx.cuda.md4.benchmarks libintx.cuda.md.eri4)
  target_link_libraries(libintx.cuda.md4.benchmarks ${libintx_test_reference})
  # tests
  add_executable(libintx.cuda.md4.test EXCLUDE_FROM_ALL libintx.cuda.md4.test.cc)
  target_link_libraries(libintx.cuda.md4.test libintx.cuda.md.eri4)
  target_link_libraries(libintx.cuda.md4.test ${libintx_test_reference})
endif()

add_executable(libintx.md.test EXCLUDE_FROM_ALL libintx.md.test.cc)
target_link_libraries(libintx.md.test libboys)
add_test(NAME libintx.md COMMAND libintx.md.test)
