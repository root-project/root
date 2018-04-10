############################################################################
# CMakeLists.txt file for building TMVA/DNN/RNN tests.
# @author Saurav Shekhar
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

set(Libraries Core MathCore Matrix TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

# RNN - BackPropagation Reference
ROOT_EXECUTABLE(testRecurrentBackpropagation TestRecurrentBackpropagation.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-RNN-Backpropagation COMMAND testRecurrentBackpropagation)
#
## RNN - Initialization Reference
#ROOT_EXECUTABLE(testRecurrentNetInit TestRecurrentNetInitialization.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-RNN-Init COMMAND testRecurrentNetInit)

# RNN - Forward Reference
ROOT_EXECUTABLE(testForwardPass TestForwardPass.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-RNN-Forward COMMAND testForwardPass)

# RNN - Full Test Reference
ROOT_EXECUTABLE(testFullRNN TestFullRNN.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-RNN-FullRNN COMMAND testFullRNN)

# RNN - Loss Reference
#ROOT_EXECUTABLE(testRecurrentNetLoss TestRecurrentNetLoss.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-RNN-Loss COMMAND testRecurrentNetLoss)
#
## RNN - Prediction Reference
#ROOT_EXECUTABLE(testRecurrentNetPred TestRecurrentNetPrediction.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-RNN-Pred COMMAND testRecurrentNetPred)

#--- CUDA tests. ---------------------------
if (CUDA_FOUND)

  SET(DNN_CUDA_LIBRARIES dnn_cuda ${CUDA_CUBLAS_LIBRARIES})

endif (CUDA_FOUND)

#--- CPU tests. ----------------------------
if (BLAS_FOUND AND imt)
  include_directories(SYSTEM ${TBB_INCLUDE_DIRS})

  # DNN - Forward CPU
  ROOT_EXECUTABLE(testForwardPassCpu TestForwardPassCpu.cxx LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-RNN-Forward-Cpu COMMAND testForwardPassCpu)

endif (BLAS_FOUND AND imt)
