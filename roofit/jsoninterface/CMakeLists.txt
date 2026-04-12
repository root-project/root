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

if(NOT CMAKE_VERSION VERSION_LESS "3.23.0")
  target_sources(
      RooFitJSONInterface
      PRIVATE
      FILE_SET private_header_files
      TYPE HEADERS
      BASE_DIRS inc src
      FILES
          src/RYMLParser.h
          src/JSONParser.h
          # LinkDef.h # is empty, not being used in ROOT_STL_PACKAGE call
          inc/RooFit/Detail/JSONInterface.h
  )
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
