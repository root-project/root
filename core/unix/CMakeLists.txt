############################################################################
# CMakeLists.txt file for building ROOT core/unix package
############################################################################

set(Unix_dict_headers
  TUnixSystem.h
)

ROOT_OBJECT_LIBRARY(Unix
  src/TUnixSystem.cxx
  G__Unix.cxx
)

target_include_directories(Unix PRIVATE ../clib/res)

ROOT_GENERATE_DICTIONARY(G__Unix
  ${Unix_dict_headers}
  STAGE1
  MULTIDICT
  MODULE
    Core
  LINKDEF
    LinkDef.h
)

ROOT_INSTALL_HEADERS()
