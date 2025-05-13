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
    RooFitHS3/JSONIO.h
    RooFitHS3/RooJSONFactoryWSTool.h
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

ROOT_ADD_TEST_SUBDIRECTORY(test)
