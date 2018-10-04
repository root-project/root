############################################################################
# CMakeLists.txt file for building ROOT tree/forest package
# @author Jakob Blomer CERN
############################################################################

if(root7)
  ROOT_GLOB_HEADERS(dictHeaders v7/inc/ROOT/*.hxx)

  ROOT_GLOB_SOURCES(sources v7/src/*.cxx)

  ROOT_STANDARD_LIBRARY_PACKAGE(ROOTForest
                                HEADERS ${dictHeaders}
                                SOURCES ${sources}
                                DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                                DEPENDENCIES RIO)

  ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
endif()
