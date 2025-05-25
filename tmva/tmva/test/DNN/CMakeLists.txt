# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA/DNN tests.
# @author Simon Pfreundschuh
############################################################################

set(Libraries TMVA)

#--- CUDA tests. ---------------------------
if (tmva-gpu)

  include_directories(${CUDA_INCLUDE_DIRS})


  SET(DNN_CUDA_LIBRARIES ${CUDA_CUBLAS_LIBRARIES} )

  list(APPEND Libraries ${DNN_CUDA_LIBRARIES} CUDA::cuda_driver CUDA::cudart)

  # DNN - Activation Functions Cuda
  add_executable(testActivationFunctionsCuda TestActivationFunctionsCuda.cxx)
  TARGET_LINK_LIBRARIES(testActivationFunctionsCuda ${Libraries} )
  ROOT_ADD_TEST(TMVA-DNN-ActivationFunctionsCuda COMMAND testActivationFunctionsCuda RESOURCE_LOCK GPU)

  # DNN - Loss Functions Cuda
  add_executable(testLossFunctionsCuda TestLossFunctionsCuda.cxx)
  TARGET_LINK_LIBRARIES(testLossFunctionsCuda ${Libraries} )
  ROOT_ADD_TEST(TMVA-DNN-LossFunctionsCuda COMMAND testLossFunctionsCuda RESOURCE_LOCK GPU)

  # DNN - Derivatives Cuda
  add_executable(testDerivativesCuda TestDerivativesCuda.cxx)
  TARGET_LINK_LIBRARIES(testDerivativesCuda ${Libraries} )
  ROOT_ADD_TEST(TMVA-DNN-DerivativesCuda COMMAND testDerivativesCuda RESOURCE_LOCK GPU)

  # DNN - Backpropagation Cuda
  add_executable(testBackpropagationCuda TestBackpropagationCuda.cxx)
  TARGET_LINK_LIBRARIES(testBackpropagationCuda ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-BackpropagationCuda COMMAND testBackpropagationCuda RESOURCE_LOCK GPU)

  # DNN - Backpropagation DL Cuda
  add_executable(testBackpropagationDLCuda TestBackpropagationDLCuda.cxx )
  TARGET_LINK_LIBRARIES(testBackpropagationDLCuda ${Libraries} )
  ROOT_ADD_TEST(TMVA-DNN-Backpropagation-DLCuda COMMAND testBackpropagationDLCuda RESOURCE_LOCK GPU)

  # DNN - Batch normalization Cuda
  add_executable(testBatchNormalizationCuda TestBatchNormalizationCuda.cxx )
  TARGET_LINK_LIBRARIES(testBatchNormalizationCuda ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-BatchNormalization-Cuda COMMAND testBatchNormalizationCuda RESOURCE_LOCK GPU)




  # DNN - Minimization Cuda
  add_executable(testMinimizationCuda TestMinimizationCuda.cxx)
  TARGET_LINK_LIBRARIES(testMinimizationCuda ${Libraries}  )
  ROOT_ADD_TEST(TMVA-DNN-MinimizationCuda COMMAND testMinimizationCuda RESOURCE_LOCK GPU)

  # DNN - Arithmetic Cuda
  add_executable(testArithmeticCuda TestMatrixArithmeticCuda.cxx)
  TARGET_LINK_LIBRARIES(testArithmeticCuda ${Libraries} )
  ROOT_ADD_TEST(TMVA-DNN-ArithmeticCuda COMMAND testArithmeticCuda RESOURCE_LOCK GPU)

  # DNN - DataLoader Cuda
  add_executable(testDataLoaderCuda TestDataLoaderCuda.cxx)
  TARGET_LINK_LIBRARIES(testDataLoaderCuda ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-DataLoaderCuda COMMAND testDataLoaderCuda RESOURCE_LOCK GPU)


  # DNN - Optimization GPU

   add_executable(testOptimizationCuda TestOptimizationCuda.cxx)
   TARGET_LINK_LIBRARIES(testOptimizationCuda ${Libraries}  )
   ROOT_ADD_TEST(TMVA-DNN-Optimization-Cuda COMMAND testOptimizationCuda RESOURCE_LOCK GPU)

#Cuda tests using CUDNN
if (tmva-cudnn)

   # DNN - Batch normalization Cudnn
   add_executable(testBatchNormalizationCudnn TestBatchNormalizationCudnn.cxx )
   TARGET_LINK_LIBRARIES(testBatchNormalizationCudnn ${Libraries})
   ROOT_ADD_TEST(TMVA-DNN-BatchNormalization-Cudnn COMMAND testBatchNormalizationCudnn RESOURCE_LOCK GPU)

   # DNN Optimization GPU Cudnn

   add_executable(testOptimizationCudnn TestOptimizationCudnn.cxx)
   TARGET_LINK_LIBRARIES(testOptimizationCudnn ${Libraries}  )
   ROOT_ADD_TEST(TMVA-DNN-Optimization-Cudnn COMMAND testOptimizationCudnn RESOURCE_LOCK GPU)

   # DNN - TensorDataLoader Cudnn
   #add_executable(testTensorDataLoaderCudnn TestTensorDataLoaderCudnn.cxx)
   #TARGET_LINK_LIBRARIES(testTensorDataLoaderCudnn ${Libraries} ${DNN_CUDA_LIBRARIES})
   #ROOT_ADD_TEST(TMVA-DNN-TensorDataLoaderCudnn COMMAND testTensorDataLoaderCudnn RESOURCE_LOCK GPU)

endif()

endif ()

#--- CPU tests. ----------------------------
#
# always run the Cpu tests. If tmva-cpu is off (no Blas or no imt)
# they will work using TMatrix operations

# DNN - Arithmetic Functions CPU
ROOT_EXECUTABLE(testArithmeticCpu TestMatrixArithmeticCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-Arithmetic-Cpu COMMAND testArithmeticCpu)

# DNN - Activation Functions CPU
ROOT_EXECUTABLE(testActivationFunctionsCpu TestActivationFunctionsCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-Activation-Functions-Cpu COMMAND testActivationFunctionsCpu)

# DNN - Loss Functions CPU
ROOT_EXECUTABLE(testLossFunctionsCpu TestLossFunctionsCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-Loss-Functions-Cpu COMMAND testLossFunctionsCpu)

# DNN - Derivatives CPU
ROOT_EXECUTABLE(testDerivativesCpu TestDerivativesCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-Derivatives-Cpu COMMAND testDerivativesCpu)

# DNN - Backpropagation CPU
ROOT_EXECUTABLE(testBackpropagationCpu TestBackpropagationCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-Backpropagation-Cpu COMMAND testBackpropagationCpu)

# DNN - BackpropagationDL CPU
ROOT_EXECUTABLE(testBackpropagationDLCpu TestBackpropagationDLCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-Backpropagation-DL-Cpu COMMAND testBackpropagationDLCpu)

# DNN - Batch normalization
ROOT_EXECUTABLE(testBatchNormalizationCpu TestBatchNormalizationCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-BatchNormalization-Cpu COMMAND testBatchNormalizationCpu)


# DNN - Optimization CPU
ROOT_EXECUTABLE(testOptimizationCpu TestOptimizationCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-Optimization-Cpu COMMAND testOptimizationCpu)

# DNN - MethodDL SGD Optimization CPU
ROOT_EXECUTABLE(testMethodDLSGDOptimizationCpu TestMethodDLSGDOptimizationCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-MethodDL-SGD-Optimization-Cpu COMMAND testMethodDLSGDOptimizationCpu)

# DNN - MethodDL Adam Optimization CPU
ROOT_EXECUTABLE(testMethodDLAdamOptimizationCpu TestMethodDLAdamOptimizationCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-MethodDL-Adam-Optimization-Cpu COMMAND testMethodDLAdamOptimizationCpu TIMEOUT 1800)

# DNN - MethodDL Adagrad Optimization CPU
ROOT_EXECUTABLE(testMethodDLAdagradOptimizationCpu TestMethodDLAdagradOptimizationCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-MethodDL-Adagrad-Optimization-Cpu COMMAND testMethodDLAdagradOptimizationCpu)

# DNN - MethodDL RMSProp Optimization CPU
ROOT_EXECUTABLE(testMethodDLRMSPropOptimizationCpu TestMethodDLRMSPropOptimizationCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-MethodDL-RMSProp-Optimization-Cpu COMMAND testMethodDLRMSPropOptimizationCpu)

# DNN - MethodDL Adadelta Optimization CPU
ROOT_EXECUTABLE(testMethodDLAdadeltaOptimizationCpu TestMethodDLAdadeltaOptimizationCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-MethodDL-Adadelta-Optimization-Cpu COMMAND testMethodDLAdadeltaOptimizationCpu)

# DNN - Regression CPU
ROOT_EXECUTABLE(testRegressionCpu TestRegressionMethodDL.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-Regression-Cpu COMMAND testRegressionCpu)

#( old-dnn-test )
# DNN - DataLoader CPU
ROOT_EXECUTABLE(testDataLoaderCpu TestDataLoaderCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-Data-Loader-Cpu COMMAND testDataLoaderCpu)

# DNN - Minimization CPU
ROOT_EXECUTABLE(testMinimizationCpu TestMinimizationCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-Minimization-Cpu COMMAND testMinimizationCpu)


# tests using TReference architecture
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



