# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Danilo Piparo CERN, 2018

ROOT_ADD_GTEST(simple simple.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testWorkspace testWorkspace.cxx LIBRARIES RooFitCore RooFit RooStats)
if(NOT MSVC OR win_broken_tests)
  ROOT_ADD_GTEST(testRooDataHist testRooDataHist.cxx LIBRARIES RooFitCore)
endif()
ROOT_ADD_GTEST(testRooWrapperPdf testRooWrapperPdf.cxx LIBRARIES Gpad RooFitCore)
ROOT_ADD_GTEST(testGenericPdf testGenericPdf.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testRooProxy testRooProxy.cxx LIBRARIES RooFitCore WILLFAIL)

