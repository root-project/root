# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

ROOT_EXECUTABLE(testRooFit testRooFit.cxx LIBRARIES RooFitCore RooFit)
ROOT_ADD_TEST(test-fit-testRooFit COMMAND testRooFit)

if(TARGET MathMore)
  ROOT_EXECUTABLE(testFitPerf testFitPerf.cxx LIBRARIES RooFitCore RooFit MathMore)
  ROOT_ADD_TEST(test-fit-testFitPerf COMMAND testFitPerf LABELS longtest)
endif()
