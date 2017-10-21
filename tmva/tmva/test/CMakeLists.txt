############################################################################
# CMakeLists.txt file for building ROOT TMVA tests.
# @author Simon Pfreundschuh
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

include_directories(${ROOT_INCLUDE_DIRS})

# Recursive addition of tests
ROOT_ADD_TEST_SUBDIRECTORY(DNN)

# For google tests
ROOT_ADD_GTEST(testCrossEvaluationSplitting TestCrossEvaluationSplitting.cxx LIBRARIES TMVA)

# For non-google tests
set(Libraries Core MathCore TMVA)

ROOT_EXECUTABLE(testCrossEvaluationSerialise TestCrossEvaluationSerialise.cxx
    LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-CrossEvaluation-Serialise COMMAND testCrossEvaluationSerialise)