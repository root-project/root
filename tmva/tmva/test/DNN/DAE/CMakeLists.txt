############################################################################
# CMakeLists.txt file for building TMVA/DNN tests.
# @author Simon Pfreundschuh
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)


set(Libraries Core MathCore Matrix TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

ROOT_EXECUTABLE(testDenoiseAEProp TestDenoiseAEProp.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-DAE-DenoiseAEProp COMMAND testDenoiseAEProp)



#--- CUDA tests. ---------------------------
if (CUDA_FOUND)
endif (CUDA_FOUND)

#--- CPU tests. ----------------------------
if (BLAS_FOUND AND imt)
endif (BLAS_FOUND AND imt)
