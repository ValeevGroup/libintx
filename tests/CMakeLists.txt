add_executable(boys.test EXCLUDE_FROM_ALL boys.test.cc)
target_link_libraries(boys.test libboys)
add_test(NAME boys COMMAND boys.test)

add_executable(interpolate.test EXCLUDE_FROM_ALL interpolate.test.cc)
target_link_libraries(interpolate.test libboys)
add_test(NAME interpolate COMMAND interpolate.test)

if (LIBINTX_ENABLE_CUDA)
  add_executable(boys.cuda.test EXCLUDE_FROM_ALL boys.cuda.test.cu)
  target_link_libraries(boys.cuda.test libboys)
  add_test(NAME boys.cuda COMMAND boys.cuda.test)
endif()
