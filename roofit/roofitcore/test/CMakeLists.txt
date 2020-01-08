# @author Patrick Bos, NL eScience Center, 2018
# @author Danilo Piparo, CERN, 2018

# Only Google Test tests are added here, the other tests are added elsewhere
ROOT_ADD_GTEST(simple simple.cxx LIBRARIES RooFitCore)
#ROOT_ADD_GTEST(testRooGradMinimizer RooGradMinimizer.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testBidirMMapPipe testBidirMMapPipe.cxx LIBRARIES RooFitCore m)
ROOT_ADD_GTEST(testMPFEnll MultiProcess/MPFEnll.cpp LIBRARIES RooFitCore)

#find_package(ZeroMQ REQUIRED)

#ROOT_ADD_GTEST(testMultiProcessVector MultiProcess_Vector.cxx LIBRARIES RooFitCore ${ZeroMQ_LIBRARY})
#ROOT_ADD_GTEST(testVectorNLL MultiProcess/VectorNLL.cpp LIBRARIES RooFitCore RooFit RooStats m ${ZeroMQ_LIBRARY})
#ROOT_ADD_GTEST(testVectorGradMinimizer MultiProcess/VectorGradMinimizer.cpp LIBRARIES RooFitCore ${ZeroMQ_LIBRARY})
