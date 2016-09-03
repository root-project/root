############################################################################
# CMakeLists.txt file for building ROOT core/lzo package
############################################################################
#
# inspired by @zzxuanyuan's pull request https://github.com/root-mirror/root/pull/81

#---Declare ZipLZO sources as part of libCore-------------------------------
set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipLZO.c)

ROOT_ADD_C_FLAG(CMAKE_C_FLAGS -std=c11)

include_directories(${LZO_INCLUDE_DIR})

ROOT_OBJECT_LIBRARY(Lzo ${sources})

