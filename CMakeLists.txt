cmake_minimum_required(VERSION 3.1)

if (NOT CMAKE_VERSION VERSION_LESS "3.13")
  # CMP0077: option() honors normal variables
  # https://cmake.org/cmake/help/latest/policy/CMP0077.html
  cmake_policy(SET CMP0077 NEW)
endif()

# set the project name
project(libintx)

set(LIBINTX_MAX_L 3 CACHE STRING "")
if (NOT DEFINED LIBINTX_MAX_X)
  math(EXPR LIBINTX_MAX_X "${LIBINTX_MAX_L}+1")
  set(LIBINTX_MAX_X ${LIBINTX_MAX_X} CACHE STRING "LIBINTX_MAX_X")
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
install(FILES ${PROJECT_BINARY_DIR}/libintx/config.h DESTINATION libintx)

include_directories(src include include/eigen3)
include_directories(${PROJECT_BINARY_DIR})

add_subdirectory(src/libintx)
add_subdirectory(src/libintx/boys)

option(LIBINTX_ENABLE_CUDA "" OFF)
if (LIBINTX_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 11 REQUIRED)

  set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "-I")
  # set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
  # set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
  # set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

  set(CMAKE_CUDA_FLAGS
"${CMAKE_CUDA_FLAGS} \
-fconcepts \
-Wall \
--expt-relaxed-constexpr \
--expt-extended-lambda \
--ptxas-options=--verbose \
-keep --source-in-ptx -lineinfo \
--objdir-as-tempdir"
  )

  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CMAKE_CUDA_ARCHITECTURES")
  endif()
  if (NOT DEFINED LIBINTX_CUDA_MAX_SHMEM)
    set(LIBINTX_CUDA_MAX_SHMEM 49152 CACHE STRING "LIBINTX_CUDA_MAX_SHMEM")
  endif()
  mark_as_advanced(LIBINTX_CUDA_MAX_SHMEM)

  add_subdirectory(src/libintx/gpu)
endif()

add_executable(pure-harmonics-table EXCLUDE_FROM_ALL src/libintx/pure-harmonics-table.cc)

add_subdirectory(src/libintx/engine/rysq/)

option(LIBINTX_LIBINT2 "" OFF)
if (LIBINTX_LIBINT2)
  find_package(Libint2 REQUIRED)
  add_subdirectory(src/libintx/engine/libint2)
endif()

option(LIBINTX_PYTHON "Python bindings" OFF)
if (LIBINTX_PYTHON)
   add_subdirectory(python)
endif()

include(CTest)
if (BUILD_TESTING)
  add_subdirectory(tests)
endif()
