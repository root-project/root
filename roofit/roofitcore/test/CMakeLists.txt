# Only Google Test tests are added here, the other tests are added elsewhere
ROOT_ADD_GTEST(testRooGradMinimizer RooGradMinimizer.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testMultiProcess_Vector MultiProcess_Vector.cxx LIBRARIES RooFitCore m)
