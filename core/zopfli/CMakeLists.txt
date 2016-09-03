############################################################################
# CMakeLists.txt file for building ROOT core/zopfli package
############################################################################
#
# inspired by @zzxuanyuan's pull request https://github.com/root-mirror/root/pull/81

#---Declare ZipZOPFLI sources as part of libCore-------------------------------
set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipZOPFLI.c)

include_directories(${ZOPFLI_INCLUDE_DIR})

ROOT_OBJECT_LIBRARY(Zopfli ${sources})

