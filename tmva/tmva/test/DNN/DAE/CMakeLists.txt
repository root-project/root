############################################################################
# CMakeLists.txt file for building TMVA/DNN/CNN tests.
# @author Akshay Vashistha
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

set(Libraries Core MathCore Matrix TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

ROOT_EXECUTABLE(testCorruptionLayer TestCorruptionLayer.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-DAE-CorruptionLayer COMMAND testCorruptionLayer)

ROOT_EXECUTABLE(testCompressionLayer TestCompressionLayer.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-DAE-CompressionLayer COMMAND testCompressionLayer)

ROOT_EXECUTABLE(testReconstructionLayer TestReconstructionLayer.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-DAE-ReconstructionLayer COMMAND testReconstructionLayer)

ROOT_EXECUTABLE(testLogisticRegressionLayer TestLogisticRegressionLayer.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-DAE-LogisticRegressionLayer COMMAND testLogisticRegressionLayer)

ROOT_EXECUTABLE(testNetPreTrain TestNetPreTrain.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-DAE-NetPreTrain COMMAND testNetPreTrain)
