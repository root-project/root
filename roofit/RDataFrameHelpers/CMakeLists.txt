# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt for registering a RooFit RDataFrameHelper
############################################################################

# This library enables compatibility of RooFit and RDataFrame.
# Since these two packages don't know about each other, we connect them here.

if(NOT dataframe)
  return()
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RooFitRDataFrameHelpers
  HEADERS
    RooAbsDataHelper.h
  NO_SOURCES
  LINKDEF
    LinkDef.h
  DEPENDENCIES
    ROOTDataFrame
    RooFitCore
)

ROOT_ADD_TEST_SUBDIRECTORY(test)


