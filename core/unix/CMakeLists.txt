############################################################################
# CMakeLists.txt file for building ROOT core/unix package
############################################################################

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)

ROOT_GLOB_HEADERS(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/T*.h)
set(Unix_dict_headers ${headers} PARENT_SCOPE)

include_directories(${CMAKE_SOURCE_DIR}/net/net/inc
  ${CMAKE_SOURCE_DIR}/core/clib/res)
ROOT_OBJECT_LIBRARY(Unix ${sources})

ROOT_INSTALL_HEADERS()
