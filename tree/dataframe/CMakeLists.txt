############################################################################
# CMakeLists.txt file for building ROOT tree/dataframe package
# @author Danilo Piparo CERN, Pere Mato CERN
############################################################################

ROOT_GLOB_HEADERS(dictHeaders inc/*.h inc/ROOT/*.hxx)

# these headers are deprecated
list(REMOVE_ITEM dictHeaders ${CMAKE_SOURCE_DIR}/tree/dataframe/inc/ROOT/TResultProxy.hxx)
list(REMOVE_ITEM dictHeaders ${CMAKE_SOURCE_DIR}/tree/dataframe/inc/ROOT/TDataFrame.hxx)

ROOT_GLOB_SOURCES(sources src/*.cxx)

set(DATAFRAME_DEPENDENCIES Imt)

if(NOT ARROW_FOUND)
  list(REMOVE_ITEM dictHeaders ${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT/RArrowDS.hxx)
  list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/RArrowDS.cxx)
endif()

if(NOT sqlite)
  list(REMOVE_ITEM dictHeaders ${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT/RSqliteDS.hxx)
  list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/RSqliteDS.cxx)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTDataFrame
                              HEADERS ${dictHeaders}
                              SOURCES ${sources}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LIBRARIES ${TBB_LIBRARIES}
                              DEPENDENCIES Tree TreePlayer Hist RIO ROOTVecOps Imt
                              ${TREEPLAYER_DEPENDENCIES})
if (ARROW_FOUND)
  target_include_directories(ROOTDataFrame ${ARROW_INCLUDE_DIR})
  target_link_libraries(ROOTDataFrame PRIVATE ${ARROW_SHARED_LIB})
endif()

if (sqlite)
  target_include_directories(ROOTDataFrame PRIVATE ${SQLITE_INCLUDE_DIR})
  target_link_libraries(ROOTDataFrame PRIVATE ${SQLITE_LIBRARIES})
  if (davix)
    target_link_libraries(ROOTDataFrame PRIVATE Davix::Davix)
  endif()
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
