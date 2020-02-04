add_executable(boys.test EXCLUDE_FROM_ALL boys.test.cc)
target_link_libraries(boys.test libboys)
add_test(NAME boys COMMAND boys.test)

add_executable(interpolate.test EXCLUDE_FROM_ALL interpolate.test.cc)
target_link_libraries(interpolate.test libboys)
add_test(NAME interpolate COMMAND interpolate.test)
