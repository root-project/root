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
    ROOT/RActionBase.hxx
    ROOT/RCsvDS.hxx
    ROOT/RCustomColumnBase.hxx
    ROOT/RCutFlowReport.hxx
    ROOT/RDataFrame.hxx
    ROOT/RDataSource.hxx
    ROOT/RDFActionHelpers.hxx
    ROOT/RDFAction.hxx
    ROOT/RDFBookedCustomColumns.hxx
    ROOT/RDFColumnValue.hxx
    ROOT/RDFCustomColumn.hxx
    ROOT/RDFDisplay.hxx
    ROOT/RDFFilter.hxx
    ROOT/RDFGraphUtils.hxx
    ROOT/RDFHelpers.hxx
    ROOT/RDFHistoModels.hxx
    ROOT/RDFInterface.hxx
    ROOT/RDFInterfaceUtils.hxx
    ROOT/RDFNodesUtils.hxx
    ROOT/RDFRange.hxx
    ROOT/RDFUtils.hxx
    ROOT/RFilterBase.hxx
    ROOT/RJittedAction.hxx
    ROOT/RJittedCustomColumn.hxx
    ROOT/RJittedFilter.hxx
    ROOT/RLazyDS.hxx
    ROOT/RLazyDSImpl.hxx
    ROOT/RLoopManager.hxx
    ROOT/RNodeBase.hxx
    ROOT/RRangeBase.hxx
    ROOT/RResultPtr.hxx
    ROOT/RRootDS.hxx
    ROOT/RSlotStack.hxx
    ROOT/RSnapshotOptions.hxx
    ROOT/RTrivialDS.hxx
    ${RDATAFRAME_EXTRA_HEADERS}
  SOURCES
    src/RActionBase.cxx
    src/RColumnValue.cxx
    src/RCsvDS.cxx
    src/RCustomColumnBase.cxx
    src/RCutFlowReport.cxx
    src/RDataFrame.cxx
    src/RDFActionHelpers.cxx
    src/RDFBookedCustomColumns.cxx
    src/RDFDisplay.cxx
    src/RDFGraphUtils.cxx
    src/RDFHistoModels.cxx
    src/RDFInterfaceUtils.cxx
    src/RDFUtils.cxx
    src/RFilterBase.cxx
    src/RJittedAction.cxx
    src/RJittedCustomColumn.cxx
    src/RJittedFilter.cxx
    src/RLoopManager.cxx
    src/RRangeBase.cxx
    src/RRootDS.cxx
    src/RSlotStack.cxx
    src/RTrivialDS.cxx
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
