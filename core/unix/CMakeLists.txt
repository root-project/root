############################################################################
# CMakeLists.txt file for building ROOT core/unix package
############################################################################

set(Unix_dict_headers
  TUnixSystem.h
  PARENT_SCOPE
)

ROOT_OBJECT_LIBRARY(Unix
  src/TUnixSystem.cxx
)

target_include_directories(Unix PRIVATE ../clib/res)

ROOT_INSTALL_HEADERS()
