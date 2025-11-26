# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT rootfit/histfactory package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RooFitCodegen
  HEADERS
    RooFit/CodegenImpl.h
  SOURCES
    src/CodegenImpl.cxx
  DICTIONARY_OPTIONS
    "-writeEmptyRootPCM"
  DEPENDENCIES
    RooFit
    RooFitCore
    HistFactory
)

if(NOT CMAKE_VERSION VERSION_LESS "3.23.0")
  target_sources(
      RooFitCodegen
      PRIVATE
      FILE_SET private_header_files
      TYPE HEADERS
      BASE_DIRS inc/
      FILES
          # LinkDef.h # empty, not being used by ROOT_STL_PACKAGE call
          inc/RooFit/CodegenImpl.h
  )
endif()
