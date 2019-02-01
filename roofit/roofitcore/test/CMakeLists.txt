# @author Danilo Piparo CERN, 2018

ROOT_ADD_GTEST(simple simple.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testWorkspace testWorkspace.cxx LIBRARIES RooFitCore RooFit RooStats)
ROOT_ADD_GTEST(testRooDataHist testRooDataHist.cxx LIBRARIES RooFitCore)
