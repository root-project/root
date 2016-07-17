############################################################################
# CMakeLists.txt file for building TMVA/DNN tests.
# @author Simon Pfreundschuh
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

set(Libraries Core MathCore Matrix TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

# DNN - Backpropagation
ROOT_EXECUTABLE(testBackpropagation TestBackpropagation.cxx
                LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-Backpropagation COMMAND testBackpropagation)

# DNN - Activation Functions
ROOT_EXECUTABLE(testActivationFunctions TestActivationFunctions.cxx
                LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-ActivationFunctions COMMAND testActivationFunctions)

# DNN - Loss Functions
ROOT_EXECUTABLE(testLossFunctions TestLossFunctions.cxx
                LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-LossFunctions COMMAND testLossFunctions)

# DNN - Derivatives
ROOT_EXECUTABLE(testDerivatives TestDerivatives.cxx
                LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-Derivatives COMMAND testDerivatives)

# DNN - Minimization
ROOT_EXECUTABLE(testMinimization TestMinimization.cxx
                LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-Minimization COMMAND testMinimization)

# DNN - Data Loader
ROOT_EXECUTABLE(testDataLoader TestDataLoader.cxx
                LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-DataLoader COMMAND testDataLoader)

find_package(CUDA)
if (CUDA_FOUND)
  # DNN - Cuda
  CUDA_ADD_EXECUTABLE(testCuda TestCuda.cxx)
  TARGET_LINK_LIBRARIES(testCuda ${Libraries} dnn_cuda ${CUDA_CUBLAS_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-CUDA COMMAND testCuda)

  # DNN - Activation Functions Cuda
  CUDA_ADD_EXECUTABLE(testActivationFunctionsCuda TestActivationFunctionsCuda.cxx)
  TARGET_LINK_LIBRARIES(testActivationFunctionsCuda ${Libraries} dnn_cuda
    ${CUDA_CUBLAS_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-ActivationFunctionsCuda COMMAND testActivationFunctionsCuda)

  # DNN - Loss Functions Cuda
  CUDA_ADD_EXECUTABLE(testLossFunctionsCuda TestLossFunctionsCuda.cxx)
  TARGET_LINK_LIBRARIES(testLossFunctionsCuda ${Libraries} dnn_cuda
    ${CUDA_CUBLAS_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-LossFunctionsCuda COMMAND testLossFunctionsCuda)

  # DNN - Derivatives Cuda
  CUDA_ADD_EXECUTABLE(testDerivativesCuda TestDerivativesCuda.cxx)
  TARGET_LINK_LIBRARIES(testDerivativesCuda ${Libraries} dnn_cuda
    ${CUDA_CUBLAS_LIBRARIES} dnn_cuda)
  ROOT_ADD_TEST(TMVA-DNN-DerivativesCuda COMMAND testDerivativesCuda)

  # DNN - Backpropagation Cuda
  CUDA_ADD_EXECUTABLE(testBackpropagationCuda TestBackpropagationCuda.cxx)
  TARGET_LINK_LIBRARIES(testBackpropagationCuda ${Libraries} dnn_cuda
    ${CUDA_CUBLAS_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-BackpropagationCuda COMMAND testBackpropagationCuda)

  # DNN - Minimization Cuda
  CUDA_ADD_EXECUTABLE(testMinimizationCuda TestMinimizationCuda.cxx)
  TARGET_LINK_LIBRARIES(testMinimizationCuda ${Libraries} dnn_cuda
    ${CUDA_CUBLAS_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-MinimizationCuda COMMAND testMinimizationCuda)
endif (CUDA_FOUND)