############################################################################
# CMakeLists.txt file for building ROOT tree/forest package
# @author Jakob Blomer CERN
############################################################################

if(root7)
  ROOT_STANDARD_LIBRARY_PACKAGE(ROOTForest
  HEADERS
    ROOT/RColumn.hxx
    ROOT/RColumnElement.hxx
    ROOT/RColumnModel.hxx
    ROOT/RForestUtil.hxx
    ROOT/RPage.hxx
    ROOT/RPagePool.hxx
    ROOT/RPageStorage.hxx
    ROOT/RPageStorageRoot.hxx
    ROOT/RTree.hxx
    ROOT/RTreeEntry.hxx
    ROOT/RTreeField.hxx
    ROOT/RTreeModel.hxx
    ROOT/RTreeValue.hxx
    ROOT/RTreeView.hxx
  SOURCES
    RColumn.cxx
    RPagePool.cxx
    RPageStorage.cxx
    RPageStorageRoot.cxx
    RTree.cxx
    RTreeField.cxx
    RTreeModel.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    RIO
  )

  ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
endif()
