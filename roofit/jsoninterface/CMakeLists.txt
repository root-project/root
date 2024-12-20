# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT RooFitJSONInterface package
# @author Jonas Rembser, CERN
############################################################################

# If RapidYAML can be found on the system, we will also compile the RapidYAML backend besides the
# nlohmann-json backend. Like this we can also convert to yaml.

# The RapidYAML backend is always disabled because it doesn't work anymore for
# RooFitHS3. The nlohmann_json interface is always used and works well, hence
# rapidyaml was not tested anymore. The relevant code is still left in the
# CMakeLists.txt in case someone wants to revive the RapilYAML backend.

# message(STATUS "Looking for RapidYAML (used by RooFit)")
# find_package(ryml)

if(${RYML_FOUND})
  message(STATUS "RapidYAML found, compiling also RooFit JSON Interface with RapidYAML parser")
  set(ParserSources src/JSONParser.cxx src/RYMLParser.cxx)
  add_compile_definitions(ROOFIT_WITH_RYML)
else()
  set(ParserSources src/JSONParser.cxx)
  message(STATUS "RapidYAML not found, only compiling RooFit with nlohmann-json parser")
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RooFitJSONInterface
  HEADERS
    RooFit/Detail/JSONInterface.h
  SOURCES
    src/JSONInterface.cxx
    ${ParserSources}
  DICTIONARY_OPTIONS
    "-writeEmptyRootPCM"
  LIBRARIES
    Core
)

if(${RYML_FOUND})
  target_include_directories(RooFitJSONInterface PRIVATE ${RYML_INCLUDE_DIRS})
  target_link_libraries(RooFitJSONInterface PRIVATE -lc4core -lryml)
  target_link_directories(RooFitJSONInterface PRIVATE ${RYML_LIB_DIR})
endif()

if(builtin_nlohmannjson)
  target_include_directories(RooFitJSONInterface PRIVATE ${CMAKE_SOURCE_DIR}/builtins)
else()
  target_link_libraries(RooFitJSONInterface PRIVATE nlohmann_json::nlohmann_json)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
