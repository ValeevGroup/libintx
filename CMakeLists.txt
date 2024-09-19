cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# CMP0077: option() honors normal variables
# https://cmake.org/cmake/help/latest/policy/CMP0077.html
cmake_policy(SET CMP0077 NEW)

# set the project name
project(libintx)

set(LIBINTX_MAX_L 3 CACHE STRING "")
if (NOT DEFINED LIBINTX_MAX_X)
  math(EXPR LIBINTX_MAX_X "${LIBINTX_MAX_L}+1")
  set(LIBINTX_MAX_X ${LIBINTX_MAX_X} CACHE STRING "LIBINTX_MAX_X")
endif()

set(LIBINTX_MAX_K 10 CACHE STRING "")

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)

option(CMAKE_POSITION_INDEPENDENT_CODE "" ON)

#find_package(Eigen3 REQUIRED)
#set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/eigen)

include_directories(src include include/eigen3)
include_directories(${PROJECT_BINARY_DIR})

#set(BLA_VENDOR Intel10_64lp_seq)
find_package(LAPACK)
if (TARGET LAPACK::LAPACK)
  message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")
  foreach (_lib in ${LAPACK_LIBRARIES})
    if (_lib MATCHES "Accelerate.framework" AND APPLE AND NOT DEFINED LIBINTX_APPLE_ACCELERATE)
      set(LIBINTX_APPLE_ACCELERATE TRUE CACHE BOOL "")
      break()
    endif()
    if (_lib MATCHES "mkl_intel" AND NOT DEFINED LIBINTX_INTEL_MKL)
      set(LIBINTX_INTEL_MKL TRUE CACHE BOOL "")
      break()
    endif()
  endforeach()
  unset(_lib)
endif()

set(LIBINTX_CBLAS_H "" CACHE STRING "")
set(LIBINTX_LAPACKE_H "" CACHE STRING "")
mark_as_advanced(LIBINTX_CBLAS_H LIBINTX_LAPACKE_H)

add_subdirectory(src/libintx)
add_subdirectory(src/libintx/boys)

option(LIBINTX_CUDA "Compile GPU/CUDA libraries " OFF)
option(LIBINTX_HIP "Compile GPU/HIP libraries" OFF)

configure_file(src/libintx/config.h.in ${PROJECT_BINARY_DIR}/libintx/config.h)
install(FILES ${PROJECT_BINARY_DIR}/libintx/config.h DESTINATION libintx)

if (LIBINTX_CUDA OR LIBINTX_HIP)
  if (LIBINTX_CUDA AND LIBINTX_HIP)
    message(FATAL_ERROR "CUDA and HIP are mutually exclusive options")
  endif()
  add_subdirectory(src/libintx/gpu)
endif()

option(LIBINTX_PYTHON "Python bindings" OFF)
if (LIBINTX_PYTHON)
   add_subdirectory(python)
endif()

include(CTest)
if (BUILD_TESTING)
  add_subdirectory(tests)
endif()
