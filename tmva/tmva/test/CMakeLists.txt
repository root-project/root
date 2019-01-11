############################################################################
# CMakeLists.txt file for building ROOT TMVA tests.
# @author Kim Albertsson
############################################################################

# Tests using google test
ROOT_ADD_GTEST(TestRandomGenerator
               TestRandomGenerator.cxx
               LIBRARIES TMVA)
ROOT_ADD_GTEST(rtensor rtensor.cxx LIBRARIES Core ROOTVecOps TMVA)
ROOT_ADD_GTEST(rtensor-utils rtensor_utils.cxx LIBRARIES Core ROOTVecOps ROOTDataFrame TMVA)
