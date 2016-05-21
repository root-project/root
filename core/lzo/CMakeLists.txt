############################################################################
# CMakeLists.txt file for building ROOT core/lzo package
############################################################################

#---Declare ZipLZO sources as part of libCore-------------------------------
set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipLZO.c)

include_directories(${LZO_INCLUDE_DIR})

ROOT_OBJECT_LIBRARY(Lzo ${sources})

