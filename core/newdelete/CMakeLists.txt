############################################################################
# CMakeLists.txt file for building ROOT core/newdelete package
############################################################################

ROOT_LINKER_LIBRARY(New
  src/MemCheck.cxx
  src/NewDelete.cxx
  DEPENDENCIES
    Core
)

target_include_directories(New PRIVATE ../clib/res)

ROOT_INSTALL_HEADERS()
