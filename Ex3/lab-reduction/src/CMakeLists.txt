
#set(SOURCES test_reduction.cu)
# cuda_compile_ptx(
#   cuda_ptx_files
#   ${SOURCES}
#   OPTIONS --source-in-ptx
# )
# add_custom_target(ptx ALL
#   DEPENDS ${cuda_ptx_files}
#   SOURCES ${SOURCES}
# )

cuda_add_executable(reduction reduction.cu)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reduction-solution.cu")
  message(STATUS "Build reduction-solution target.")
  cuda_add_executable(reduction-solution reduction-solution.cu)
  target_link_libraries(reduction-solution PRIVATE Common)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reduction-solution-n.cu")
  message(STATUS "Build reduction-solution-n target.")
  cuda_add_executable(reduction-solution-n reduction-solution-n.cu)
  target_link_libraries(reduction-solution-n PRIVATE Common)
endif()

# Requires cub library
# git clone https://github.com/NVlabs/cub.git ext/
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reduction-lib.cu")
  message(STATUS "Build reduction-lib target.")
  cuda_add_executable(reduction-lib reduction-lib.cu)
  target_include_directories(reduction-lib PRIVATE "${CMAKE_SOURCE_DIR}/ext/")
  target_link_libraries(reduction-lib PRIVATE Common)
endif()
