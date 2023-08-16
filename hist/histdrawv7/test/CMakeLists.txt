# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

ROOT_ADD_GTEST(drawUnit draw.cxx LIBRARIES ROOTHistDraw)
# On Windows, this test receives "unexpected diagnostic of severity 2000"
# because "pointer was truncated (due a missing dictionary)" and then trying
# to allocate "an object of abstract class type 'RHistImplBase'".
if(NOT MSVC OR win_broken_tests)
  ROOT_ADD_GTEST(ioUnit io.cxx LIBRARIES ROOTHistDraw)
endif()
