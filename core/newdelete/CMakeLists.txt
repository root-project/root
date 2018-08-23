############################################################################
# CMakeLists.txt file for building ROOT core/newdelete package
############################################################################

set(sources src/MemCheck.cxx
   src/NewDelete.cxx
)
include_directories(${CMAKE_SOURCE_DIR}/core/clib/res)

ROOT_LINKER_LIBRARY(New ${sources} DEPENDENCIES Core)
ROOT_INSTALL_HEADERS()
