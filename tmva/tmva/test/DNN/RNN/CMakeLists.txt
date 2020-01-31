# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA/DNN/RNN tests.
# @author Saurav Shekhar
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

set(Libraries Core MathCore Matrix TMVA)

if (CUDA_FOUND AND  tmva-gpu)
include_directories(${ROOT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
else()
include_directories(${ROOT_INCLUDE_DIRS})
endif()

#---Reference test

if (Test_Reference)

# RNN - BackPropagation Reference
ROOT_EXECUTABLE(testRecurrentBackpropagation TestRecurrentBackpropagation.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-RNN-Backpropagation COMMAND testRecurrentBackpropagation)
#
## RNN - Initialization Reference
#ROOT_EXECUTABLE(testRecurrentNetInit TestRecurrentNetInitialization.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-RNN-Init COMMAND testRecurrentNetInit)

# RNN - Forward Reference
ROOT_EXECUTABLE(testRecurrentForwardPass TestRecurrentForwardPass.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-RNN-Forward COMMAND testRecurrentForwardPass)

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

endif()

#--- CUDA tests. ---------------------------
if (CUDA_FOUND AND tmva-gpu)

set(DNN_CUDA_LIBRARIES ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES} )

CUDA_ADD_EXECUTABLE(testRecurrentBackpropagationCuda TestRecurrentBackpropagationCuda.cxx)
target_link_libraries(testRecurrentBackpropagationCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
ROOT_ADD_TEST(TMVA-DNN-RNN-BackpropagationCuda COMMAND testRecurrentBackpropagationCuda)
#mark teh native Cuda test failing since RNN are supported only with Cudnn
set_tests_properties(TMVA-DNN-RNN-BackpropagationCuda PROPERTIES WILL_FAIL true)

CUDA_ADD_EXECUTABLE(testRecurrentBackpropagationCudnn TestRecurrentBackpropagationCudnn.cxx)
target_link_libraries(testRecurrentBackpropagationCudnn ${Libraries} ${DNN_CUDA_LIBRARIES})
ROOT_ADD_TEST(TMVA-DNN-RNN-BackpropagationCudnn COMMAND testRecurrentBackpropagationCudnn)

endif()

#--- CPU tests. ----------------------------
if ((BLAS_FOUND OR mathmore) AND imt AND tmva-cpu)
  include_directories(${TBB_INCLUDE_DIRS})

  # DNN - Forward CPU
  ROOT_EXECUTABLE(testRecurrentForwardPassCpu TestRecurrentForwardPassCpu.cxx LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-RNN-Forward-Cpu COMMAND testRecurrentForwardPassCpu)

 ROOT_EXECUTABLE(testRecurrentBackpropagationCpu TestRecurrentBackpropagationCpu.cxx LIBRARIES ${Libraries})
 ROOT_ADD_TEST(TMVA-DNN-RNN-Backpropagation-Cpu COMMAND testRecurrentBackpropagationCpu)

# RNN - Full Test Reference
ROOT_EXECUTABLE(testFullRNNCpu TestFullRNNCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-RNN-FullRNN-Cpu COMMAND testFullRNNCpu)


endif ()
