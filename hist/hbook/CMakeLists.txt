############################################################################
# CMakeLists.txt file for building ROOT hist/hbook package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Hbook
  HEADERS
    THbookBranch.h
    THbookFile.h
    THbookKey.h
    THbookTree.h
  SOURCES
    src/THbookBranch.cxx
    src/THbookFile.cxx
    src/THbookKey.cxx
    src/THbookTree.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  LIBRARIES
    minicern
  DEPENDENCIES
    Hist
    Matrix
    Tree
    Graf
    TreePlayer
    RIO
)
