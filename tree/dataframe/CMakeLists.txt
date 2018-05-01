############################################################################
# CMakeLists.txt file for building ROOT tree/dataframe package
# @author Danilo Piparo CERN, Pere Mato CERN
############################################################################

if (ARROW_FOUND)
  include_directories(${ARROW_INCLUDE_DIR})
endif()

ROOT_GLOB_HEADERS(dictHeaders inc/*.h inc/ROOT/*.hxx)

# this header is deprecated
list(REMOVE_ITEM dictHeaders ${CMAKE_SOURCE_DIR}/tree/dataframe/inc/ROOT/TResultProxy.hxx)

ROOT_GLOB_SOURCES(sources src/*.cxx)

set(DATAFRAME_DEPENDENCIES Imt)

if(NOT ARROW_FOUND)
  list(REMOVE_ITEM dictHeaders ${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT/TArrowDS.hxx)
  list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/TArrowDS.cxx)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(DataFrame
                              HEADERS ${dictHeaders}
                              SOURCES ${sources}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LIBRARIES ${TBB_LIBRARIES} ${ARROW_SHARED_LIB}
                              DEPENDENCIES Tree TreePlayer Hist RIO VecOps Imt
                              ${TREEPLAYER_DEPENDENCIES})

ROOT_ADD_TEST_SUBDIRECTORY(test)
