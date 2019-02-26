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
    v7/inc/ROOT/RForestUtil.hxx
    v7/inc/ROOT/RPage.hxx
    v7/inc/ROOT/RPagePool.hxx
    v7/inc/ROOT/RPageStorage.hxx
    v7/inc/ROOT/RPageStorageRoot.hxx
    v7/inc/ROOT/RTreeEntry.hxx
    v7/inc/ROOT/RTreeField.hxx
    v7/inc/ROOT/RTreeModel.hxx
    v7/inc/ROOT/RTreeValue.hxx
    v7/inc/ROOT/RTreeView.hxx
  SOURCES
    v7/src/RColumn.cxx
    v7/src/RForest.cxx
    v7/src/RPage.cxx
    v7/src/RPagePool.cxx
    v7/src/RPageStorage.cxx
    v7/src/RPageStorageRoot.cxx
    v7/src/RTreeEntry.cxx
    v7/src/RTreeField.cxx
    v7/src/RTreeModel.cxx
  LINKDEF
    LinkDef.h
  DEPENDENCIES
    RIO
  )

  ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
endif()
