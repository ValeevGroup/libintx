cmake_minimum_required(VERSION 3.1)

# set the project name
project(libintx)

set(LIBINTX_MAX_L 3 CACHE STRING "")
if (NOT LIBINTX_MAX_X)
  math(EXPR LIBINTX_MAX_X "${LIBINTX_MAX_L}+1")
  #set(LIBINTX_MAX_X ${LIBINTX_MAX_X} CACHE FORCE STRING "")
endif()

set(LIBINTX_MAX_K 10 CACHE STRING "")

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)

option(CMAKE_POSITION_INDEPENDENT_CODE "" ON)

#find_package(Eigen3 REQUIRED)
#set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/eigen)

configure_file(src/libintx/config.h.in ${PROJECT_BINARY_DIR}/libintx/config.h)

include_directories(src include)
include_directories(${PROJECT_BINARY_DIR})

include(CTest)
if (BUILD_TESTING)
  add_subdirectory(tests)
endif()
