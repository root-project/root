############################################################################
# CMakeLists.txt file for building ROOT tree/forest package
# @author Jakob Blomer CERN
############################################################################

if(root7)
  ROOT_STANDARD_LIBRARY_PACKAGE(ROOTForest
  HEADERS
    ROOT/RBranch.hxx
    ROOT/RColumn.hxx
    ROOT/RColumnElement.hxx
    ROOT/RForestUtil.hxx
    ROOT/RPage.hxx
    ROOT/RPagePool.hxx
    ROOT/RPageStorage.hxx
    ROOT/RTree.hxx
    ROOT/RTreeEntry.hxx
    ROOT/RTreeModel.hxx
    ROOT/RTreeValue.hxx
    ROOT/RTreeView.hxx
  SOURCES
    RBranch.cxx
    RPageStorage.cxx
    RTree.cxx
    RTreeModel.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    RIO
  )

  ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
endif()
