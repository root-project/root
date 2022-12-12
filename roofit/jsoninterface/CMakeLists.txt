# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT RooFitJSONInterface package
# @author Jonas Rembser, CERN
############################################################################

if(roofit_hs3_ryml)
  # If RapidYAML can be found on the system, we will use RapidYAML instead of
  # nlohmann-json. Like this we can also convert to yaml.

  message(STATUS "Looking for RapidYAML (used by RooFit)")
  find_package(ryml)
  if(${RYML_FOUND})
    message(STATUS "RapidYAML found, compiling RooFit JSON Interface with RapidYAML parser")
    set(ParserSource src/RYMLParser.cxx)
    add_compile_definitions(ROOFIT_WITH_RYML)
  else()
    set(ParserSource src/JSONParser.cxx)
    message(STATUS "RapidYAML not found, compiling RooFit with nlohmann-json parser")
  endif()
else()
  set(ParserSource src/JSONParser.cxx)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RooFitJSONInterface
  HEADERS
    RooFit/Detail/JSONInterface.h
  SOURCES
    src/JSONInterface.cxx
    ${ParserSource}
  DICTIONARY_OPTIONS
    "-writeEmptyRootPCM"
  LIBRARIES
    Core
)

if((roofit_hs3_ryml) AND (${RYML_FOUND}))
  target_include_directories(RooFitJSONInterface PRIVATE ${RYML_INCLUDE_DIRS})
  target_link_libraries(RooFitJSONInterface PRIVATE -lc4core -lryml)
  target_link_directories(RooFitJSONInterface PRIVATE ${RYML_LIB_DIR})
else()
  if(builtin_nlohmannjson)
    target_include_directories(RooFitJSONInterface PRIVATE ${CMAKE_SOURCE_DIR}/builtins)
  else()
    target_link_libraries(RooFitJSONInterface PUBLIC nlohmann_json::nlohmann_json)
  endif()
endif()
