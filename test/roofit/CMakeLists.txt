ROOT_EXECUTABLE(testRooFit testRooFit.cxx LIBRARIES RooFitCore RooFit)
ROOT_ADD_TEST(test-roofit-testRooFit COMMAND testRooFit)

if(TARGET MathMore)
  ROOT_EXECUTABLE(testFitPerf testFitPerf.cxx LIBRARIES RooFitCore RooFit MathMore)
  ROOT_ADD_TEST(test-roofit-testFitPerf COMMAND testFitPerf LABELS longtest)
endif()
