############################################################################
# CMakeLists.txt file for building ROOT tree/dataframe package
# @author Danilo Piparo CERN, Pere Mato CERN
############################################################################

if (ARROW_FOUND)
  include_directories(${ARROW_INCLUDE_DIR})
endif()

if (sqlite)
  include_directories(${SQLITE_INCLUDE_DIR})
  set (DATAFRAME_SQLITE_LIBRARIES ${SQLITE_LIBRARIES})
endif()

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
                              LIBRARIES ${TBB_LIBRARIES} ${ARROW_SHARED_LIB} ${DATAFRAME_SQLITE_LIBRARIES}
                              DEPENDENCIES Tree TreePlayer Hist RIO ROOTVecOps Imt
                              ${TREEPLAYER_DEPENDENCIES})

ROOT_ADD_TEST_SUBDIRECTORY(test)
