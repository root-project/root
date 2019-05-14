############################################################################
# CMakeLists.txt file for building ROOT forest package
# @author Jakob Blomer CERN
############################################################################

if(NOT root7)
  return()
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTForest
HEADERS
  ROOT/RColumn.hxx
  ROOT/RColumnElement.hxx
  ROOT/RColumnModel.hxx
  ROOT/RField.hxx
  ROOT/RFieldValue.hxx
  ROOT/RForest.hxx
  ROOT/RForestDescriptor.hxx
  ROOT/RForestEntry.hxx
  ROOT/RForestModel.hxx
  ROOT/RForestUtil.hxx
  ROOT/RForestView.hxx
  ROOT/RPage.hxx
  ROOT/RPagePool.hxx
  ROOT/RPageStorage.hxx
  ROOT/RPageStorageRoot.hxx
SOURCES
  v7/src/RColumn.cxx
  v7/src/RField.cxx
  v7/src/RForest.cxx
  v7/src/RForestDescriptor.cxx
  v7/src/RForestEntry.cxx
  v7/src/RForestModel.cxx
  v7/src/RPage.cxx
  v7/src/RPagePool.cxx
  v7/src/RPageStorage.cxx
  v7/src/RPageStorageRoot.cxx
LINKDEF
  LinkDef.h
DEPENDENCIES
  RIO
  ROOTVecOps
)

ROOT_ADD_TEST_SUBDIRECTORY(v7/test)

