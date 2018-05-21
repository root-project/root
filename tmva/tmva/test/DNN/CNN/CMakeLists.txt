############################################################################
# CMakeLists.txt file for building TMVA/DNN/CNN tests.
# @author Vladimir Ilievski
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

set(Libraries Core MathCore Matrix TMVA)

if (CUDA_FOUND AND tmva-gpu)
include_directories(${ROOT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
else()
include_directories(${ROOT_INCLUDE_DIRS})
endif()

ROOT_EXECUTABLE(testIm2Col TestIm2Col.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Im2Col COMMAND testIm2Col)

ROOT_EXECUTABLE(testDownsample TestDownsample.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Downsample COMMAND testDownsample)

ROOT_EXECUTABLE(testFlatten TestFlatten.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-testFlatten COMMAND testFlatten)

ROOT_EXECUTABLE(testRotWeights TestRotateWeights.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-RotWeights COMMAND testRotWeights)

#-- features not implemeted in ref architectures 
#ROOT_EXECUTABLE(testForwardPass TestForwardPass.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-CNN-Forward COMMAND testForwardPass)

#ROOT_EXECUTABLE(testConvNetLoss TestConvNetLoss.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-CNN-Loss COMMAND testConvNetLoss)

#ROOT_EXECUTABLE(testConvNetPred TestConvNetPrediction.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-CNN-Pred COMMAND testConvNetPred)

#ROOT_EXECUTABLE(testDLMinimization TestMinimization.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-CNN-Minimization COMMAND testDLMinimization)

#ROOT_EXECUTABLE(testTensorDataLoader TestTensorDataLoader.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-Tensor-Data-Loader COMMAND testTensorDataLoader)

#--- CUDA tests. ---------------------------
if (CUDA_FOUND)

    SET(DNN_CUDA_LIBRARIES dnn_cuda ${CUDA_CUBLAS_LIBRARIES})

    # DNN - Activation Functions Cuda
    CUDA_ADD_EXECUTABLE(testIm2ColCuda TestIm2ColCuda.cxx)
    TARGET_LINK_LIBRARIES(testIm2ColCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
    ROOT_ADD_TEST(TMVA-DNN-CNN-Im2ColCuda COMMAND testIm2ColCuda)

endif (CUDA_FOUND)

#--- CPU tests. ----------------------------
if ((BLAS_FOUND OR mathmore) AND imt AND tmva-cpu)

include_directories(SYSTEM ${TBB_INCLUDE_DIRS})

ROOT_EXECUTABLE(testIm2ColCpu TestIm2ColCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Im2Col-CPU COMMAND testIm2ColCpu)

ROOT_EXECUTABLE(testDownsampleCpu TestDownsampleCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Downsample-CPU COMMAND testDownsampleCpu)

ROOT_EXECUTABLE(testRotWeightsCpu TestRotateWeightsCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-RotWeights-CPU COMMAND testRotWeightsCpu)

ROOT_EXECUTABLE(testForwardPassCpu TestForwardPassCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Forward-CPU COMMAND testForwardPassCpu)

ROOT_EXECUTABLE(testConvNetLossCpu TestConvNetLossCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Loss-CPU COMMAND testConvNetLossCpu)

ROOT_EXECUTABLE(testConvNetPredCpu TestConvNetPredictionCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Pred-CPU COMMAND testConvNetPredCpu)

#-- need to be fixed 
#ROOT_EXECUTABLE(testTensorDataLoaderCpu TestTensorDataLoaderCpu.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-Tensor-Data-Loader-CPU COMMAND testTensorDataLoaderCpu)

ROOT_EXECUTABLE(testDLMinimizationCpu TestMinimizationCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Minimization-CPU COMMAND testDLMinimizationCpu)

ROOT_EXECUTABLE(testConvBackpropagationCpu TestConvBackpropagation.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-Backpropagation-CPU COMMAND testConvBackpropagationCpu)


ROOT_EXECUTABLE(testMethodDLCpu TestMethodDL.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-MethodDL-CPU COMMAND testMethodDLCpu)


endif ()
