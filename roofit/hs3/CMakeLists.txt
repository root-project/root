# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT rootfit/histfactory package
# @author Carsten Burgard, CERN
############################################################################

set (ROOFITHS3_HEADERS
  RooFitHS3/JSONIO.h
  RooFitHS3/RooJSONFactoryWSTool.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(RooFitHS3
  HEADERS
    ${ROOFITHS3_HEADERS}
  SOURCES
    src/Domains.cxx
    src/JSONIO.cxx
    src/RooJSONFactoryWSTool.cxx
    src/JSONFactories_RooFitCore.cxx
    src/JSONFactories_HistFactory.cxx
    src/JSONIOUtils.cxx
  DICTIONARY_OPTIONS
    "-writeEmptyRootPCM"
  DEPENDENCIES
    HistFactory
    RooFit
    RooFitCore
    RooFitJSONInterface
)

set(RELATIVE_INC_HEADERS ${ROOFITHS3_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
if(NOT CMAKE_VERSION VERSION_LESS "3.23.0")
  target_sources(
      RooFitHS3
      PRIVATE
      FILE_SET private_header_files
      TYPE HEADERS
      BASE_DIRS inc/ src/
      FILES
          ${RELATIVE_INC_HEADERS}
          # LinkDef.h # seemingly not being used by ROOT_STL_PACKAGE call
          src/Domains.h
          src/JSONIOUtils.h
          src/static_execute.h
  )
endif()


ROOT_ADD_TEST_SUBDIRECTORY(test)
