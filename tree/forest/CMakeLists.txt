############################################################################
# CMakeLists.txt file for building ROOT forest package
# @author Jakob Blomer CERN
############################################################################

if(root7)
  ROOT_STANDARD_LIBRARY_PACKAGE(ROOTForest
  HEADERS
    v7/inc/ROOT/RColumn.hxx
    v7/inc/ROOT/RColumnElement.hxx
    v7/inc/ROOT/RColumnModel.hxx
    v7/inc/ROOT/RForest.hxx
    v7/inc/ROOT/RForestEntry.hxx
    v7/inc/ROOT/RForestModel.hxx
    v7/inc/ROOT/RForestUtil.hxx
    v7/inc/ROOT/RForestView.hxx
    v7/inc/ROOT/RPage.hxx
    v7/inc/ROOT/RPagePool.hxx
    v7/inc/ROOT/RPageStorage.hxx
    v7/inc/ROOT/RPageStorageRoot.hxx
    v7/inc/ROOT/RTreeField.hxx
    v7/inc/ROOT/RTreeValue.hxx
  SOURCES
    v7/src/RColumn.cxx
    v7/src/RForest.cxx
    v7/src/RForestEntry.cxx
    v7/src/RForestModel.cxx
    v7/src/RPage.cxx
    v7/src/RPagePool.cxx
    v7/src/RPageStorage.cxx
    v7/src/RPageStorageRoot.cxx
    v7/src/RTreeField.cxx
  LINKDEF
    LinkDef.h
  DEPENDENCIES
    RIO
  )

  ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
endif()
