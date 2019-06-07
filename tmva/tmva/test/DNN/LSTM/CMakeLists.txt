############################################################################
# CMakeLists.txt file for building TMVA/DNN/LSTM tests.
# @author Harshit Prasad
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

set(Libraries Core MathCore Matrix TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

# LSTM - Forward Reference
ROOT_EXECUTABLE(testLSTMForwardPass TestLSTMForwardPass.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-LSTM-Forward COMMAND testLSTMForwardPass)

# LSTM - Backpropagation Reference
#ROOT_EXECUTABLE(testLSTMBackpropagation TestLSTMBackpropagation.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-LSTM-Backpropagation COMMAND testLSTMBackpropagation)

#--- CUDA tests. ---------------------------
if (CUDA_FOUND)

    SET(DNN_CUDA_LIBRARIES dnn_cuda ${CUDA_CUBLAS_LIBRARIES})

endif (CUDA_FOUND)

#--- CPU tests. ----------------------------
if (BLAS_FOUND AND imt)
    include_directories(SYSTEM ${TBB_INCLUDE_DIRS})

    # LSTM - Forward CPU
    ROOT_EXECUTABLE(testLSTMForwardPassCpu TestLSTMForwardPassCpu.cxx LIBRARIES ${Libraries})
    ROOT_ADD_TEST(TMVA-DNN-LSTM-Forward-Cpu COMMAND testLSTMForwardPassCpu)

    # LSTM - Backward CPU
    #ROOT_EXECUTABLE(testLSTMBackpropagationCpu TestLSTMBackpropagationCpu.cxx LIBRARIES ${Libraries})
    #ROOT_ADD_TEST(TMVA-DNN-LSTM-Backward-Cpu COMMAND testLSTMBackpropagationCpu)

endif (BLAS_FOUND AND imt)
