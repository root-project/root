############################################################################
# CMakeLists.txt file for building ROOT core/lz4 package
############################################################################
#
# inspired by @zzxuanyuan's pull request https://github.com/root-mirror/root/pull/81

#---Declare ZipLZ4 sources as part of libCore-------------------------------
if(builtin_lz4)
  set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipLZ4.c ${CMAKE_CURRENT_SOURCE_DIR}/src/builtin/lz4.c)
  ROOT_ADD_C_FLAG(CMAKE_C_FLAGS -DBUILTIN_LZ4)
else()
  set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipLZ4.c)
  include_directories(${LZ4_INCLUDE_DIR})
endif()

ROOT_ADD_C_FLAG(CMAKE_C_FLAGS -std=c11)

ROOT_OBJECT_LIBRARY(Lz4 ${sources})

