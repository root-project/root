############################################################################
# CMakeLists.txt file for building TMVA/DNN tests.
# @author Simon Pfreundschuh
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

set(Libraries Core MathCore Matrix TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

#--- CUDA tests. ---------------------------
if (CUDA_FOUND)

  SET(DNN_CUDA_LIBRARIES dnn_cuda ${CUDA_CUBLAS_LIBRARIES})

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
endif (CUDA_FOUND)

#--- CPU tests. ----------------------------
if (BLAS_FOUND AND imt)
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

  # DNN - DataLoader CPU
  ROOT_EXECUTABLE(testDataLoaderCpu TestDataLoaderCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Data-Loader-Cpu COMMAND testDataLoaderCpu)

  # DNN - Minimization CPU
  ROOT_EXECUTABLE(testMinimizationCpu TestMinimizationCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Minimization-Cpu COMMAND testMinimizationCpu)

endif (BLAS_FOUND AND imt)
