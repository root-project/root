# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

if(NOT minimal)
    ROOT_ADD_GTEST(TTabComTests TTabComTests.cxx LIBRARIES Rint)
endif()
ROOT_ADD_GTEST(TRintTests TRintTests.cxx LIBRARIES Rint)

