# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT rootfit/histfactory package
# @author Carsten Burgard, CERN
############################################################################

# If it can be found on the system, we will use rapidyaml instead of
# nlohmann-json. Like this we can also convert to yaml.

message(STATUS "Looking for RYML (used by RooFitHS3)")
find_package(ryml)
if(${RYML_FOUND})
  message(STATUS "RYML found, compiling RooFitHS3 with RYAML parser")
else()
  message(STATUS "RYML not found, compiling RooFitHS3 with JSON parser")
endif()

if(${RYML_FOUND})
    set(RYMLSources src/RYMLParser.cxx)
    add_compile_definitions(ROOFIT_HS3_WITH_RYML)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RooFitHS3
  HEADERS
    RooFitHS3/RooJSONFactoryWSTool.h
    RooFitHS3/HistFactoryJSONTool.h
  SOURCES
    src/RooJSONFactoryWSTool.cxx
    src/HistFactoryJSONTool.cxx
    src/JSONInterface.cxx
    src/JSONFactories_RooFitCore.cxx
    src/JSONFactories_HistFactory.cxx
    src/JSONParser.cxx
    ${RYMLSources}
  DICTIONARY_OPTIONS
    "-writeEmptyRootPCM"
  DEPENDENCIES
    RooFit
    RooFitCore
    RooStats
    HistFactory
)

if(${RYML_FOUND})
  target_include_directories(RooFitHS3 PRIVATE ${RYML_INCLUDE_DIRS})
  target_link_libraries(RooFitHS3 PRIVATE -lc4core -lryml)
  target_link_directories(RooFitHS3 PRIVATE ${RYML_LIB_DIR})
else()
  target_include_directories(RooFitHS3 PRIVATE ${CMAKE_SOURCE_DIR}/io/io/res)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
