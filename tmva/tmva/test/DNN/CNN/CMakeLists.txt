############################################################################
# CMakeLists.txt file for building TMVA/DNN tests.
# @author Simon Pfreundschuh
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)


set(Libraries Core MathCore Matrix TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

ROOT_EXECUTABLE(testConvBackpropagation TestConvBackpropagation.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Backpropagation COMMAND testConvBackpropagation)

ROOT_EXECUTABLE(testIm2Col TestIm2Col.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Im2Col COMMAND testIm2Col)

ROOT_EXECUTABLE(testDownsample TestDownSample.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Downsample COMMAND testDownsample)

ROOT_EXECUTABLE(testRotWeights TestRotateWeights.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-rotWeights COMMAND testrotWeights)

ROOT_EXECUTABLE(testFlatten TestFlatten.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-testFlatten COMMAND testFlatten)

ROOT_EXECUTABLE(testConvNetInit TestConvNetInitialization.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Init COMMAND testConvNetInit)

ROOT_EXECUTABLE(testForwardPass TestForwardPass.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Forward COMMAND testForwardPass)

ROOT_EXECUTABLE(testConvNetLoss TestConvNetLoss.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Loss COMMAND testConvNetLoss)

ROOT_EXECUTABLE(testConvNetPred TestConvNetPrediction.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Pred COMMAND testConvNetPred)


#--- CUDA tests. ---------------------------
if (CUDA_FOUND)
endif (CUDA_FOUND)

#--- CPU tests. ----------------------------
if (BLAS_FOUND AND imt)
endif (BLAS_FOUND AND imt)
