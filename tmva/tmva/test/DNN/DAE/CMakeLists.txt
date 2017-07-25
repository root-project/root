############################################################################
# CMakeLists.txt file for building TMVA/DNN/CNN tests.
# @author Akshay Vashistha
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

set(Libraries Core MathCore Matrix TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

ROOT_EXECUTABLE(testDenoiseLayer TestDenoiseLayer.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-DAE-DenoiseLayer COMMAND testDenoiseLayer)

ROOT_EXECUTABLE(testLogisticRegressionLayer TestLogisticRegressionLayer.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-DAE-LogisticRegressionLayer COMMAND testLogisticRegressionLayer)
