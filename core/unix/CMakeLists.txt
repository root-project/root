############################################################################
# CMakeLists.txt file for building ROOT core/unix package
############################################################################

set(sources src/TUnixSystem.cxx)

set(Unix_dict_headers  ${CMAKE_CURRENT_SOURCE_DIR}/inc/TUnixSystem.h PARENT_SCOPE)

include_directories(${CMAKE_SOURCE_DIR}/net/net/inc
  ${CMAKE_SOURCE_DIR}/core/clib/res)

ROOT_OBJECT_LIBRARY(Unix ${sources})

ROOT_INSTALL_HEADERS()
