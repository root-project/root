############################################################################
# CMakeLists.txt file for building ROOT hist/unfold package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Unfold
  HEADERS
    TUnfold.h
    TUnfoldBinning.h
    TUnfoldBinningXML.h
    TUnfoldDensity.h
    TUnfoldSys.h
  SOURCES
    src/TUnfold.cxx
    src/TUnfoldBinning.cxx
    src/TUnfoldBinningXML.cxx
    src/TUnfoldDensity.cxx
    src/TUnfoldSys.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Hist
    XMLParser
    Matrix
)
