############################################################################
# CMakeLists.txt file for building ROOT TMVA tests.
# @author Simon Pfreundschuh
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

ROOT_ADD_TEST_SUBDIRECTORY(DNN)

# For google tests
ROOT_ADD_GTEST(testCrossEvaluationSplitting TestCrossEvaluationSplitting.cxx LIBRARIES TMVA)

