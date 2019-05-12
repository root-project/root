############################################################################
# CMakeLists.txt file for building ROOT forest package
# @author Jakob Blomer CERN
############################################################################

if(NOT root7)
  return()
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTDataSet
HEADERS
  ROOT/RColumn.hxx
  ROOT/RColumnElement.hxx
  ROOT/RColumnModel.hxx
  ROOT/RDataSet.hxx
  ROOT/RDataSetDescriptor.hxx
  ROOT/RDataSetModel.hxx
  ROOT/RDataSetUtil.hxx
  ROOT/RDataSetView.hxx
  ROOT/REntry.hxx
  ROOT/RField.hxx
  ROOT/RFieldValue.hxx
  ROOT/RPage.hxx
  ROOT/RPagePool.hxx
  ROOT/RPageStorage.hxx
  ROOT/RPageStorageRoot.hxx
SOURCES
  v7/src/RColumn.cxx
  v7/src/RDataSet.cxx
  v7/src/RDataSetDescriptor.cxx
  v7/src/RDataSetModel.cxx
  v7/src/REntry.cxx
  v7/src/RField.cxx
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

