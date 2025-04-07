if(NOT root7)
  return()
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTFile
HEADERS
  ROOT/RFile.hxx
SOURCES
  src/RFile.cxx
 LINKDEF
  LinkDef.h
DEPENDENCIES
  RIO
  Tree
  Hist
)

 ROOT_ADD_TEST_SUBDIRECTORY(test)
