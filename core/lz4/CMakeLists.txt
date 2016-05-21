############################################################################
# CMakeLists.txt file for building ROOT core/lz4 package
############################################################################

#---Declare ZipLZ4 sources as part of libCore-------------------------------
if(builtin_lz4)
  set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipLZ4.c ${CMAKE_CURRENT_SOURCE_DIR}/src/lz4.c)
  ROOT_ADD_CXX_FLAG(CMAKE_C_FLAGS -DBUILTIN_LZ4)
else()
  set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipLZ4.c)
endif()

include_directories(${LZ4_INCLUDE_DIR})

ROOT_OBJECT_LIBRARY(Lz4 ${sources})

