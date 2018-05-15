ROOT_EXECUTABLE(testRooFit testRooFit.cxx LIBRARIES RooFitCore RooFit)
ROOT_ADD_TEST(test-fit-testRooFit COMMAND testRooFit)

if(TARGET MathMore)
  ROOT_EXECUTABLE(testFitPerf testFitPerf.cxx LIBRARIES RooFitCore RooFit MathMore)
  ROOT_ADD_TEST(test-fit-testFitPerf COMMAND testFitPerf LABELS longtest)
endif()
