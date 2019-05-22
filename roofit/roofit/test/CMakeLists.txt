# @author Stephan Hageboeck, CERN, 2019

ROOT_ADD_GTEST(testRooGaussian testRooGaussian.cxx LIBRARIES RooFitCore RooFit)
ROOT_ADD_GTEST(testRooPoisson testRooPoisson.cxx LIBRARIES RooFitCore RooFit)
ROOT_ADD_GTEST(testRooJohnson testRooJohnson.cxx LIBRARIES Gpad RooFitCore RooFit)
