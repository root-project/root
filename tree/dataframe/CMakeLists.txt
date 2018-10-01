############################################################################
# CMakeLists.txt file for building ROOT tree/dataframe package
# @author Danilo Piparo CERN, Pere Mato CERN
############################################################################

# Disable RDataFrame on 32-bit UNIX platforms due to ROOT-9236
if(UNIX AND CMAKE_SIZEOF_VOID_P EQUAL 4)
  return()
endif()

if(arrow)
  list(APPEND RDATAFRAME_EXTRA_HEADERS ROOT/RArrowDS.hxx)
  list(APPEND RDATAFRAME_EXTRA_INCLUDES -I${ARROW_INCLUDE_DIR})
endif()

if(sqlite)
  list(APPEND RDATAFRAME_EXTRA_HEADERS ROOT/RSqliteDS.hxx)
  list(APPEND RDATAFRAME_EXTRA_INCLUDES -I${SQLITE_INCLUDE_DIR})
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTDataFrame
  HEADERS
    ROOT/GraphNode.hxx
    ROOT/RCsvDS.hxx
    ROOT/RCutFlowReport.hxx
    ROOT/RDataFrame.hxx
    ROOT/RDataSource.hxx
    ROOT/RDFActionHelpers.hxx
    ROOT/RDFBookedCustomColumns.hxx
    ROOT/RDFDisplay.hxx
    ROOT/RDFGraphUtils.hxx
    ROOT/RDFHelpers.hxx
    ROOT/RDFHistoModels.hxx
    ROOT/RDFInterface.hxx
    ROOT/RDFInterfaceUtils.hxx
    ROOT/RDFNodes.hxx
    ROOT/RDFNodesUtils.hxx
    ROOT/RDFUtils.hxx
    ROOT/RLazyDS.hxx
    ROOT/RLazyDSImpl.hxx
    ROOT/RLoopManager.hxx
    ROOT/RNodeBase.hxx
    ROOT/RResultPtr.hxx
    ROOT/RRootDS.hxx
    ROOT/RSlotStack.hxx
    ROOT/RSnapshotOptions.hxx
    ROOT/RTrivialDS.hxx
    ${RDATAFRAME_EXTRA_HEADERS}
  SOURCES
    RCsvDS.cxx
    RCutFlowReport.cxx
    RDataFrame.cxx
    RDFActionHelpers.cxx
    RDFBookedCustomColumns.cxx
    RDFDisplay.cxx
    RDFGraphUtils.cxx
    RDFHistoModels.cxx
    RDFInterfaceUtils.cxx
    RDFNodes.cxx
    RDFUtils.cxx
    RLoopManager.cxx
    RRootDS.cxx
    RSlotStack.cxx
    RTrivialDS.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
    ${RDATAFRAME_EXTRA_INCLUDES}
  DEPENDENCIES
    Tree
    TreePlayer
    Hist
    RIO
    ROOTVecOps
    Imt
)

if(arrow)
  target_sources(ROOTDataFrame PRIVATE src/RArrowDS.cxx)
  target_include_directories(ROOTDataFrame PRIVATE ${ARROW_INCLUDE_DIR})
  target_link_libraries(ROOTDataFrame PRIVATE ${ARROW_SHARED_LIB})
endif()

if(sqlite)
  target_sources(ROOTDataFrame PRIVATE src/RSqliteDS.cxx)
  target_include_directories(ROOTDataFrame PUBLIC ${SQLITE_INCLUDE_DIR})
  target_link_libraries(ROOTDataFrame PUBLIC ${SQLITE_LIBRARIES})
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
