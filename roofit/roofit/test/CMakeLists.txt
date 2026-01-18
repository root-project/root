# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Stephan Hageboeck, CERN, 2019

ROOT_ADD_GTEST(testRooFit
    stabilityTests.cxx
    testRooBernstein.cxx
    testRooBifurGauss.cxx
    testRooCrystalBall.cxx
    testRooExponential.cxx
    testRooGaussian.cxx
    testRooJohnson.cxx
    testRooKeysPdf.cxx
    testRooLandau.cxx
    testRooParamHistFunc.cxx
    testRooPoisson.cxx
    testRooStepFunc.cxx
  LIBRARIES
    Gpad
    RooFit
  COPY_TO_BUILDDIR
    # For testRooExponential.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/exponentialPdf.root
)

if(mathmore)
  ROOT_EXECUTABLE(testRooFitExe testRooFit.cxx LIBRARIES RooFit MathMore)
  ROOT_ADD_TEST(test-fit-testRooFit COMMAND testRooFitExe)
  ROOT_EXECUTABLE(testFitPerf testFitPerf.cxx LIBRARIES RooFit MathMore)
  ROOT_ADD_TEST(test-fit-testFitPerf COMMAND testFitPerf)
else()
  ROOT_EXECUTABLE(testRooFitExe testRooFit.cxx LIBRARIES RooFit)
  ROOT_ADD_TEST(test-fit-testRooFit COMMAND testRooFitExe)
endif()

add_subdirectory(vectorisedPDFs)
