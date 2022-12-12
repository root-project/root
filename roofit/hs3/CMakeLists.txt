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
    RooFitHS3/RooJSONFactoryWSTool.h
    RooFitHS3/HistFactoryJSONTool.h
  SOURCES
    src/RooJSONFactoryWSTool.cxx
    src/HistFactoryJSONTool.cxx
    src/JSONFactories_RooFitCore.cxx
    src/JSONFactories_HistFactory.cxx
  DICTIONARY_OPTIONS
    "-writeEmptyRootPCM"
  LIBRARIES
    RooFit
    RooFitCore
    RooStats
    HistFactory
    RooFitJSONInterface
)

ROOT_ADD_TEST_SUBDIRECTORY(test)
