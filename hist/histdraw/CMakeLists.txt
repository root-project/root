############################################################################
# CMakeLists.txt file for building ROOT hist/histdraw package
############################################################################

if(NOT root7)
  return()
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTHistDraw
  HEADERS
    ROOT/RHistDrawable.hxx
    ROOT/RHistDrawingOpts.hxx
  SOURCES
    v7/src/RHistDrawable.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Hist
    ROOTGpadv7
)

ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
