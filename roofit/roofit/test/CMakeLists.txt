# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Stephan Hageboeck, CERN, 2019

ROOT_ADD_GTEST(stabilityTests stabilityTests.cxx LIBRARIES RooFit)
ROOT_ADD_GTEST(testRooBernstein testRooBernstein.cxx LIBRARIES RooFit)
ROOT_ADD_GTEST(testRooBifurGauss testRooBifurGauss.cxx LIBRARIES RooFit)
ROOT_ADD_GTEST(testRooCrystalBall testRooCrystalBall.cxx LIBRARIES Gpad RooFit)
ROOT_ADD_GTEST(testRooExponential testRooExponential.cxx
  COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/exponentialPdf.root
  LIBRARIES Core RooFit)
ROOT_ADD_GTEST(testRooGaussian testRooGaussian.cxx LIBRARIES RooFit)
ROOT_ADD_GTEST(testRooJohnson testRooJohnson.cxx LIBRARIES Gpad RooFit)
ROOT_ADD_GTEST(testRooKeysPdf testRooKeysPdf.cxx LIBRARIES Gpad RooFit)
ROOT_ADD_GTEST(testRooLandau testRooLandau.cxx LIBRARIES RooFit)
ROOT_ADD_GTEST(testRooParamHistFunc testRooParamHistFunc.cxx LIBRARIES Gpad RooFit)
ROOT_ADD_GTEST(testRooPoisson testRooPoisson.cxx LIBRARIES RooFit)
ROOT_ADD_GTEST(testRooStepFunc testRooStepFunc.cxx LIBRARIES RooFit)
ROOT_ADD_GTEST(testRooMultiPdf testRooMultiPdf.cxx LIBRARIES RooFit)
if(mathmore)
  ROOT_EXECUTABLE(testRooFit testRooFit.cxx LIBRARIES RooFit MathMore)
  ROOT_ADD_TEST(test-fit-testRooFit COMMAND testRooFit)
  ROOT_EXECUTABLE(testFitPerf testFitPerf.cxx LIBRARIES RooFit MathMore)
  ROOT_ADD_TEST(test-fit-testFitPerf COMMAND testFitPerf)
else()
  ROOT_EXECUTABLE(testRooFit testRooFit.cxx LIBRARIES RooFit)
  ROOT_ADD_TEST(test-fit-testRooFit COMMAND testRooFit)
endif()

add_subdirectory(vectorisedPDFs)
