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
endif()

if(root7)
  list(APPEND RDATAFRAME_EXTRA_HEADERS ROOT/RNTupleDS.hxx)
  list(APPEND RDATAFRAME_EXTRA_DEPS ROOTNTuple)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTDataFrame
  HEADERS
    ROOT/RCsvDS.hxx
    ROOT/RDataFrame.hxx
    ROOT/RDataSource.hxx
    ROOT/RDFHelpers.hxx
    ROOT/RLazyDS.hxx
    ROOT/RResultPtr.hxx
    ROOT/RRootDS.hxx
    ROOT/RSnapshotOptions.hxx
    ROOT/RTrivialDS.hxx
    ROOT/RDF/ActionHelpers.hxx
    ROOT/RDF/GraphNode.hxx
    ROOT/RDF/GraphUtils.hxx
    ROOT/RDF/HistoModels.hxx
    ROOT/RDF/InterfaceUtils.hxx
    ROOT/RDF/NodesUtils.hxx
    ROOT/RDF/RActionBase.hxx
    ROOT/RDF/RAction.hxx
    ROOT/RDF/RBookedCustomColumns.hxx
    ROOT/RDF/RColumnValue.hxx
    ROOT/RDF/RCustomColumnBase.hxx
    ROOT/RDF/RCustomColumn.hxx
    ROOT/RDF/RCutFlowReport.hxx
    ROOT/RDF/RDisplay.hxx
    ROOT/RDF/RFilterBase.hxx
    ROOT/RDF/RFilter.hxx
    ROOT/RDF/RInterface.hxx
    ROOT/RDF/RJittedAction.hxx
    ROOT/RDF/RJittedCustomColumn.hxx
    ROOT/RDF/RJittedFilter.hxx
    ROOT/RDF/RLazyDSImpl.hxx
    ROOT/RDF/RLoopManager.hxx
    ROOT/RDF/RNodeBase.hxx
    ROOT/RDF/RRangeBase.hxx
    ROOT/RDF/RRange.hxx
    ROOT/RDF/RSlotStack.hxx
    ROOT/RDF/Utils.hxx
    ROOT/RDF/PyROOTHelpers.hxx
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
    ${RDATAFRAME_EXTRA_DEPS}
)

if(arrow)
  target_sources(ROOTDataFrame PRIVATE src/RArrowDS.cxx)
  target_include_directories(ROOTDataFrame PRIVATE ${ARROW_INCLUDE_DIR})
  target_link_libraries(ROOTDataFrame PRIVATE ${ARROW_SHARED_LIB})
endif()

if(sqlite)
  target_sources(ROOTDataFrame PRIVATE src/RSqliteDS.cxx)
  target_include_directories(ROOTDataFrame PRIVATE ${SQLITE_INCLUDE_DIR})
  target_link_libraries(ROOTDataFrame PRIVATE ${SQLITE_LIBRARIES})
  if (davix)
    # TODO(jblomer): Use Davix through an abstraction layer in RDAVIX to avoid linking issues
    if (builtin_davix)
      # The builtin version is statically linked, thus we link against RDAVIX to avoid a symbol conflict
      # if libROOTDataFrame and libRDAVIX are used together.
      target_include_directories(ROOTDataFrame PRIVATE ${DAVIX_INCLUDE_DIRS})
      target_link_libraries(ROOTDataFrame PRIVATE RDAVIX)
    else()
      # If the system libDavix is used, RDAVIX does not contain the Davix symbols itself
      target_link_libraries(ROOTDataFrame PRIVATE Davix::Davix)
    endif()
  endif()
endif()

if(root7)
  target_sources(ROOTDataFrame PRIVATE src/RNTupleDS.cxx)
endif(root7)

ROOT_ADD_TEST_SUBDIRECTORY(test)
