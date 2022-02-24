# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT rootfit/histfactory package
# @author Carsten Burgard, CERN
############################################################################

if(roofit_hs3_ryml)
  # If RapidYAML can be found on the system, we will use RapidYAML instead of
  # nlohmann-json. Like this we can also convert to yaml.

  message(STATUS "Looking for RapidYAML (used by RooFitHS3)")
  find_package(ryml)
  if(${RYML_FOUND})
    message(STATUS "RapidYAML found, compiling RooFitHS3 with RapidYAML parser")
    set(RYMLSources src/RYMLParser.cxx)
    add_compile_definitions(ROOFIT_HS3_WITH_RYML)
  else()
    message(STATUS "RapidYAML not found, compiling RooFitHS3 with nlohmann-json parser")
  endif()
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RooFitHS3
  HEADERS
    RooFitHS3/RooJSONFactoryWSTool.h
    RooFitHS3/HistFactoryJSONTool.h
    RooFitHS3/JSONInterface.h
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
  LIBRARIES
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
  if(builtin_nlohmannjson)
    target_include_directories(RooFitHS3 PRIVATE ${CMAKE_SOURCE_DIR}/builtins)
  else()
    target_link_libraries(RooFitHS3 PUBLIC nlohmann_json::nlohmann_json)
  endif()
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
