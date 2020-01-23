############################################################################
# CMakeLists.txt file for building TMVA/DNN/GRU tests.
# @author Surya S Dwivedi
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

set(Libraries Core MathCore Matrix TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

# GRU - Forward Reference
#ROOT_EXECUTABLE(testGRUForwardPass TestGRUForwardPass.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-GRU-Forward COMMAND testGRUForwardPass)

# GRU - Backpropagation Reference
#ROOT_EXECUTABLE(testGRUBackpropagation TestGRUBackpropagation.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-GRU-Backpropagation COMMAND testGRUBackpropagation)

# GRU - Full Test Reference
#ROOT_EXECUTABLE(testFullGRU TestFullGRU.cxx LIBRARIES ${Libraries})
#ROOT_ADD_TEST(TMVA-DNN-GRU-FullGRU COMMAND testFullGRU)


#--- CUDA tests. ---------------------------
if (CUDA_FOUND)

    SET(DNN_CUDA_LIBRARIES dnn_cuda ${CUDA_CUBLAS_LIBRARIES})

endif (CUDA_FOUND)

#--- CPU tests. ----------------------------
if (BLAS_FOUND AND imt)
    include_directories(SYSTEM ${TBB_INCLUDE_DIRS})

    # GRU - Forward CPU
    ROOT_EXECUTABLE(testGRUForwardPassCpu TestGRUForwardPassCpu.cxx LIBRARIES ${Libraries})
    ROOT_ADD_TEST(TMVA-DNN-GRU-Forward-Cpu COMMAND testGRUForwardPassCpu)

    # GRU - Backward CPU
    ROOT_EXECUTABLE(testGRUBackpropagationCpu TestGRUBackpropagationCpu.cxx LIBRARIES ${Libraries})
    ROOT_ADD_TEST(TMVA-DNN-GRU-Backward-Cpu COMMAND testGRUBackpropagationCpu)

    # GRU - Full Test Reference
    ROOT_EXECUTABLE(testFullGRUCpu TestFullGRUCpu.cxx LIBRARIES ${Libraries})
    ROOT_ADD_TEST(TMVA-DNN-GRU-FullGRU-Cpu COMMAND testFullGRUCpu)

endif (BLAS_FOUND AND imt)
