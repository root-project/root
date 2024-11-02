############################################################################
# CMakeLists.txt file for building TMVA/DNN/LSTM tests.
# @author Surya S Dwivedi
############################################################################


set(Libraries TMVA)

# LSTM - Forward Reference
#ROOT_EXECUTABLE(testLSTMForwardPass TestLSTMForwardPass.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-LSTM-Forward COMMAND testLSTMForwardPass)

# LSTM - Backpropagation Reference
#ROOT_EXECUTABLE(testLSTMBackpropagation TestLSTMBackpropagation.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-LSTM-Backpropagation COMMAND testLSTMBackpropagation)

# LSTM - Full Test Reference
#ROOT_EXECUTABLE(testFullLSTM TestFullLSTM.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-LSTM-FullLSTM COMMAND testFullLSTM)


#--- CUDA tests. ---------------------------
if (tmva-gpu AND tmva-cudnn)
   list(APPEND Libraries CUDA::cuda_driver CUDA::cudart)

   set(DNN_CUDA_LIBRARIES ${CUDA_CUBLAS_LIBRARIES} )

   add_executable(testLSTMForwardPassCudnn TestLSTMForwardPassCudnn.cxx)
   target_link_libraries(testLSTMForwardPassCudnn ${Libraries} ${DNN_CUDA_LIBRARIES})
   ROOT_ADD_TEST(TMVA-DNN-LSTM-Forwaed-Cudnn COMMAND testLSTMForwardPassCudnn RESOURCE_LOCK GPU)

   add_executable(testLSTMBackpropagationCudnn TestLSTMBackpropagationCudnn.cxx)
   target_link_libraries(testLSTMBackpropagationCudnn ${Libraries} ${DNN_CUDA_LIBRARIES})
   ROOT_ADD_TEST(TMVA-DNN-LSTM-BackpropagationCudnn COMMAND testLSTMBackpropagationCudnn RESOURCE_LOCK GPU)
   # Test crashes on ubuntu2404-cuda-12.6.1. See root-project/root#16790:
   set_tests_properties(TMVA-DNN-LSTM-BackpropagationCudnn PROPERTIES DISABLED True)

   # LSTM - Full Test GPU
   add_executable(testFullLSTMCudnn TestFullLSTMCudnn.cxx)
   target_link_libraries(testFullLSTMCudnn ${Libraries} ${DNN_CUDA_LIBRARIES})
   ROOT_ADD_TEST(TMVA-DNN-LSTM-Full-Cudnn COMMAND testFullLSTMCudnn RESOURCE_LOCK GPU)

endif (tmva-gpu AND tmva-cudnn)

#--- CPU tests. ----------------------------
if (BLAS_FOUND AND imt)

    # LSTM - Forward CPU
    ROOT_EXECUTABLE(testLSTMForwardPassCpu TestLSTMForwardPassCpu.cxx LIBRARIES ${Libraries})
    ROOT_ADD_TEST(TMVA-DNN-LSTM-Forward-Cpu COMMAND testLSTMForwardPassCpu)

    # LSTM - Backward CPU
    ROOT_EXECUTABLE(testLSTMBackpropagationCpu TestLSTMBackpropagationCpu.cxx LIBRARIES ${Libraries})
    ROOT_ADD_TEST(TMVA-DNN-LSTM-Backward-Cpu COMMAND testLSTMBackpropagationCpu)

    # LSTM - Full Test Reference
    ROOT_EXECUTABLE(testFullLSTMCpu TestFullLSTMCpu.cxx LIBRARIES ${Libraries})
    ROOT_ADD_TEST(TMVA-DNN-LSTM-Full-Cpu COMMAND testFullLSTMCpu)

endif (BLAS_FOUND AND imt)
