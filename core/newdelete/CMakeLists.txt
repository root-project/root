############################################################################
# CMakeLists.txt file for building ROOT core/newdelete package
############################################################################

include_directories(${CMAKE_SOURCE_DIR}/core/clib/res)

ROOT_LINKER_LIBRARY(New *.cxx DEPENDENCIES Core)
ROOT_INSTALL_HEADERS()

