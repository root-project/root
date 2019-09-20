# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA/DNN tests.
# @author Simon Pfreundschuh
############################################################################

set(Libraries Core MathCore Matrix TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

ROOT_ADD_TEST_SUBDIRECTORY(CNN)
ROOT_ADD_TEST_SUBDIRECTORY(RNN)

#--- CUDA tests. ---------------------------
if (CUDA_FOUND AND tmva-gpu)

  include_directories(${ROOT_INCLUDE_DIRS}  ${CUDA_INCLUDE_DIRS})


  #if using CUDA_ADD_EXECUTABLE CUDA_LIBRARIES are added automatically
  SET(DNN_CUDA_LIBRARIES ${CUDA_CUBLAS_LIBRARIES} /usr/local/cuda/lib64/libcudnn.so)


  # DNN - Activation Functions Cuda
  CUDA_ADD_EXECUTABLE(testActivationFunctionsCuda TestActivationFunctionsCuda.cxx)
  TARGET_LINK_LIBRARIES(testActivationFunctionsCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-ActivationFunctionsCuda COMMAND testActivationFunctionsCuda)

  # DNN - Loss Functions Cuda
  CUDA_ADD_EXECUTABLE(testLossFunctionsCuda TestLossFunctionsCuda.cxx)
  TARGET_LINK_LIBRARIES(testLossFunctionsCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-LossFunctionsCuda COMMAND testLossFunctionsCuda)

  # DNN - Derivatives Cuda
  CUDA_ADD_EXECUTABLE(testDerivativesCuda TestDerivativesCuda.cxx)
  TARGET_LINK_LIBRARIES(testDerivativesCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-DerivativesCuda COMMAND testDerivativesCuda)

  # DNN - Backpropagation Cuda
  CUDA_ADD_EXECUTABLE(testBackpropagationCuda TestBackpropagationCuda.cxx)
  TARGET_LINK_LIBRARIES(testBackpropagationCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-BackpropagationCuda COMMAND testBackpropagationCuda)

  # DNN - Backpropagation DL Cuda
  CUDA_ADD_EXECUTABLE(testBackpropagationDLCuda TestBackpropagationDLCuda.cxx )
  TARGET_LINK_LIBRARIES(testBackpropagationDLCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-Backpropagation-DLCuda COMMAND testBackpropagationDLCuda)

  # DNN - Batch normalization Cuda
  CUDA_ADD_EXECUTABLE(testBatchNormalizationCuda TestBatchNormalizationCuda.cxx )
  TARGET_LINK_LIBRARIES(testBatchNormalizationCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-BatchNormalization-Cuda COMMAND testBatchNormalizationCuda)

  # DNN - Minimization Cuda
  CUDA_ADD_EXECUTABLE(testMinimizationCuda TestMinimizationCuda.cxx)
  TARGET_LINK_LIBRARIES(testMinimizationCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-MinimizationCuda COMMAND testMinimizationCuda)

  # DNN - Arithmetic Cuda
  CUDA_ADD_EXECUTABLE(testArithmeticCuda TestMatrixArithmeticCuda.cxx)
  TARGET_LINK_LIBRARIES(testArithmeticCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-ArithmeticCuda COMMAND testArithmeticCuda)

  # DNN - DataLoader Cuda
  CUDA_ADD_EXECUTABLE(testDataLoaderCuda TestDataLoaderCuda.cxx)
  TARGET_LINK_LIBRARIES(testDataLoaderCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-DataLoaderCuda COMMAND testDataLoaderCuda)
  
  # DNN - Cudnn
if (Have_cudnn) 
  CUDA_ADD_EXECUTABLE(testCudnn TestCudnn.cxx)
  TARGET_LINK_LIBRARIES(testCudnn ${Libraries} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-Cudnn COMMAND testCudnn)
endif()

   # DNN - Optimization GPU
  ROOT_EXECUTABLE(testOptimizationCuda TestOptimizationCuda.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Optimization-Cuda COMMAND testOptimizationCuda)

  # DNN - TensorDataLoader Cudnn
  #CUDA_ADD_EXECUTABLE(testTensorDataLoaderCudnn TestTensorDataLoaderCudnn.cxx)
  #TARGET_LINK_LIBRARIES(testTensorDataLoaderCudnn ${Libraries} ${DNN_CUDA_LIBRARIES})
  #ROOT_ADD_TEST(TMVA-DNN-TensorDataLoaderCudnn COMMAND testTensorDataLoaderCudnn)

endif ()

#--- CPU tests. ----------------------------
if ( (BLAS_FOUND OR mathmore) AND imt AND tmva-cpu)
  include_directories(${TBB_INCLUDE_DIRS})

  # DNN - Arithmetic Functions CPU
  ROOT_EXECUTABLE(testArithmeticCpu TestMatrixArithmeticCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Arithmetic-Cpu COMMAND testArithmeticCpu)

  # DNN - Activation Functions CPU
  ROOT_EXECUTABLE(testActivationFunctionsCpu TestActivationFunctionsCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Activation-Functions-Cpu COMMAND testActivationFunctionsCpu)

  # DNN - Loss Functions CPU
  ROOT_EXECUTABLE(testLossFunctionsCpu TestLossFunctionsCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Loss-Functions-Cpu COMMAND testLossFunctionsCpu)

  # DNN - Derivatives CPU
  ROOT_EXECUTABLE(testDerivativesCpu TestDerivativesCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Derivatives-Cpu COMMAND testDerivativesCpu)

  # DNN - Backpropagation CPU
  ROOT_EXECUTABLE(testBackpropagationCpu TestBackpropagationCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Backpropagation-Cpu COMMAND testBackpropagationCpu)

  # DNN - BackpropagationDL CPU
  ROOT_EXECUTABLE(testBackpropagationDLCpu TestBackpropagationDLCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Backpropagation-DL-Cpu COMMAND testBackpropagationDLCpu)

    # DNN - Batch normalization
  ROOT_EXECUTABLE(testBatchNormalizationCpu TestBatchNormalizationCpu.cxx LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-BatchNormalization-Cpu COMMAND testBatchNormalizationCpu)

if ( old-dnn-test ) 
  # DNN - DataLoader CPU
  ROOT_EXECUTABLE(testDataLoaderCpu TestDataLoaderCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Data-Loader-Cpu COMMAND testDataLoaderCpu)

  # DNN - Minimization CPU
  ROOT_EXECUTABLE(testMinimizationCpu TestMinimizationCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Minimization-Cpu COMMAND testMinimizationCpu)
  
endif ()

  

  # DNN - Optimization CPU
  ROOT_EXECUTABLE(testOptimizationCpu TestOptimizationCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Optimization-Cpu COMMAND testOptimizationCpu)

  # DNN - MethodDL SGD Optimization CPU
  ROOT_EXECUTABLE(testMethodDLSGDOptimizationCpu TestMethodDLSGDOptimizationCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-MethodDL-SGD-Optimization-Cpu COMMAND testMethodDLSGDOptimizationCpu)

  # DNN - MethodDL Adam Optimization CPU
  ROOT_EXECUTABLE(testMethodDLAdamOptimizationCpu TestMethodDLAdamOptimizationCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-MethodDL-Adam-Optimization-Cpu COMMAND testMethodDLAdamOptimizationCpu)

  # DNN - MethodDL Adagrad Optimization CPU
  ROOT_EXECUTABLE(testMethodDLAdagradOptimizationCpu TestMethodDLAdagradOptimizationCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-MethodDL-Adagrad-Optimization-Cpu COMMAND testMethodDLAdagradOptimizationCpu)

  # DNN - MethodDL RMSProp Optimization CPU
  ROOT_EXECUTABLE(testMethodDLRMSPropOptimizationCpu TestMethodDLRMSPropOptimizationCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-MethodDL-RMSProp-Optimization-Cpu COMMAND testMethodDLRMSPropOptimizationCpu)

  # DNN - MethodDL Adadelta Optimization CPU
  ROOT_EXECUTABLE(testMethodDLAdadeltaOptimizationCpu TestMethodDLAdadeltaOptimizationCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-MethodDL-Adadelta-Optimization-Cpu COMMAND testMethodDLAdadeltaOptimizationCpu)
  
  # DNN - Regression CPU
  ROOT_EXECUTABLE(testRegressionCpu TestRegressionMethodDL.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Regression-Cpu COMMAND testRegressionCpu)

endif ()

if ( reference-tests)

  # DNN - Activation Functions
  ROOT_EXECUTABLE(testActivationFunctions TestActivationFunctions.cxx LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Activation-Functions COMMAND testActivationFunctions)

  # DNN - Loss Functions
  ROOT_EXECUTABLE(testLossFunctions TestLossFunctions.cxx LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Loss-Functions COMMAND testLossFunctions)

  # DNN - Derivatives
  ROOT_EXECUTABLE(testDerivatives TestDerivatives.cxx LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Derivatives COMMAND testDerivatives)

  # DNN - Backpropagation
  ROOT_EXECUTABLE(testBackpropagation TestBackpropagation.cxx LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Backpropagation COMMAND testBackpropagation)

  # DNN - Backpropagation DL
  ROOT_EXECUTABLE(testBackpropagationDL TestBackpropagationDL.cxx LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Backpropagation-DL COMMAND testBackpropagationDL)

  # DNN - Batch normalization
  ROOT_EXECUTABLE(testBatchNormalization TestBatchNormalization.cxx LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-BatchNormalization COMMAND testBatchNormalization)

  # DNN - DataLoader
  ROOT_EXECUTABLE(testDataLoader TestDataLoader.cxx LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Data-Loader COMMAND testDataLoader)

  # DNN - Minimization
#  ROOT_EXECUTABLE(testMinimization TestMinimization.cxx LIBRARIES ${Libraries})
#  # this test takes more than 20 minutes on arm in non-optimised mode
#  if (NOT (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" AND CMAKE_BUILD_TYPE STREQUAL "Debug") )
#    ROOT_ADD_TEST(TMVA-DNN-Minimization COMMAND testMinimization)
#  endif()
endif()
