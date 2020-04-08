# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT rootfit/histfactory package
# @author Carsten Burgard, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RooFitHS3
  HEADERS
    RooFitHS3/JSONInterface.h
    RooFitHS3/RooJSONFactoryWSTool.h
    RooFitHS3/HistFactoryJSONTool.h
    RooFitHS3/JSONParser.h
    RooFitHS3/RYMLParser.h
  SOURCES
    src/RooJSONFactoryWSTool.cxx
    src/HistFactoryJSONTool.cxx
    src/JSONInterface.cxx
    src/JSONParser.cxx
    src/RYMLParser.cxx
    src/JSONFactories_RooFitCore.cxx
    src/JSONFactories_HistFactory.cxx
  DICTIONARY_OPTIONS
    "-writeEmptyRootPCM"
  DEPENDENCIES
    RooFit
    RooFitCore
    RooStats
    HistFactory    
)

if(ryml)
  target_include_directories(RooFitHS3 PRIVATE
    ${RYML_INCLUDE_DIRS}
    )
  target_link_libraries(RooFitHS3 PRIVATE -lc4core -lryml)
  target_link_directories(RooFitHS3 PRIVATE
    ${RYML_LIB_DIR}
    )
else()
  target_include_directories(RooFitHS3 PRIVATE
    ${CMAKE_SOURCE_DIR}/io/io/res
    )
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
