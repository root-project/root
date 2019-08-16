# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Simon Leisibach CERN

if(NOT dataframe)
  return()
endif()

ROOT_ADD_GTEST(ntuple_browse ntuple_browse.cxx LIBRARIES Hist ROOTNTuple ROOTNTupleBrowse)
