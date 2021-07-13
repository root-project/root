# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT RooFitJSONInterface package
# @author Jonas Rembser, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RooFitJSONInterface
  HEADERS
    RooFit/Detail/JSONInterface.h
  SOURCES
    src/JSONInterface.cxx
    src/JSONParser.cxx
  DICTIONARY_OPTIONS
    "-writeEmptyRootPCM"
  DEPENDENCIES
    Core
)

target_link_libraries(RooFitJSONInterface PRIVATE nlohmann_json::nlohmann_json)

ROOT_ADD_TEST_SUBDIRECTORY(test)
