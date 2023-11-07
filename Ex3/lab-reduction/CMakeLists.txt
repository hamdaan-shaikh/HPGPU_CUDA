cmake_minimum_required (VERSION 3.1)

# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

include(cmake/safeguards.cmake)
include(cmake/init_build_type.cmake)


project(reduction
  VERSION 0.1.1
  LANGUAGES C CXX)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "gcc5.0 or newer required.")
  endif()
endif()

# --- Common ---

add_library(Common INTERFACE)
add_library(reduction::common ALIAS Common)

target_include_directories(Common INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/inc>
)
target_compile_features(Common
  INTERFACE cxx_std_11
)
# --- CUDA ---
find_package(CUDA REQUIRED)

list(APPEND CUDA_NVCC_FLAGS "--gpu-architecture=compute_61 --gpu-code=sm_61")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "-lineinfo")

add_library(Cuda INTERFACE)
add_library(reduction::cuda ALIAS Cuda)

# workaround for FindCUDA that uses plain link_libraries
set(CUDA_LIBRARIES PRIVATE ${CUDA_LIBRARIES})


# add_subdirectory(source_dir [binary_dir] ...)
add_subdirectory(src bin)
