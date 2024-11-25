############################################################################
# CMakeLists.txt file for building TMVA/DNN/GRU tests.
# @author Surya S Dwivedi
############################################################################


set(Libraries TMVA)


#--- CUDA tests. ---------------------------
if (tmva-gpu AND tmva-cudnn)

   list(APPEND Libraries CUDA::cuda_driver CUDA::cudart)

   set(DNN_CUDA_LIBRARIES ${CUDA_CUBLAS_LIBRARIES} )

   add_executable(testGRUBackpropagationCudnn TestGRUBackpropagationCudnn.cxx)
   target_link_libraries(testGRUBackpropagationCudnn ${Libraries} ${DNN_CUDA_LIBRARIES})
   ROOT_ADD_TEST(TMVA-DNN-GRU-BackpropagationCudnn COMMAND testGRUBackpropagationCudnn RESOURCE_LOCK GPU)

   add_executable(testGRUForwardPassCudnn TestGRUForwardPassCudnn.cxx)
   target_link_libraries(testGRUForwardPassCudnn ${Libraries} ${DNN_CUDA_LIBRARIES})
   ROOT_ADD_TEST(TMVA-DNN-GRU-Forwaed-Cudnn COMMAND testGRUForwardPassCudnn RESOURCE_LOCK GPU)

   # GRU - Full Test GPU
   add_executable(testFullGRUCudnn TestFullGRUCudnn.cxx)
   target_link_libraries(testFullGRUCudnn ${Libraries} ${DNN_CUDA_LIBRARIES})
   ROOT_ADD_TEST(TMVA-DNN-GRU-Full-Cudnn COMMAND testFullGRUCudnn RESOURCE_LOCK GPU)

endif (tmva-gpu AND tmva-cudnn)

#--- CPU tests. ----------------------------
if (BLAS_FOUND AND imt)

    # GRU - Forward CPU
    ROOT_EXECUTABLE(testGRUForwardPassCpu TestGRUForwardPassCpu.cxx LIBRARIES ${Libraries})
    ROOT_ADD_TEST(TMVA-DNN-GRU-Forward-Cpu COMMAND testGRUForwardPassCpu)

    # GRU - Backward CPU
    ROOT_EXECUTABLE(testGRUBackpropagationCpu TestGRUBackpropagationCpu.cxx LIBRARIES ${Libraries})
    ROOT_ADD_TEST(TMVA-DNN-GRU-Backward-Cpu COMMAND testGRUBackpropagationCpu)

    # GRU - Full Test CPU
    ROOT_EXECUTABLE(testFullGRUCpu TestFullGRUCpu.cxx LIBRARIES ${Libraries})
    ROOT_ADD_TEST(TMVA-DNN-GRU-Full-Cpu COMMAND testFullGRUCpu)

endif (BLAS_FOUND AND imt)
