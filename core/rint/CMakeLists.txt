############################################################################
# CMakeLists.txt file for building ROOT core/rint package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Rint
  STAGE1
  HEADERS
    TRint.h
    TTabCom.h
  SOURCES
    src/TRint.cxx
    src/TTabCom.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
)
