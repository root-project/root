# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT tree/tree package
# @author Pere Mato, CERN
############################################################################

if (NOT cuda)
  return()
endif()

# ROOT_STANDARD_LIBRARY_PACKAGE(CUDAHist
#   HEADERS
#     RHnCUDA.cuh
#   # NO_SOURCES
#   SOURCES
#     RhnCUDA.cxx
#   DEPENDENCIES
#     MathCore
# )

# project(CUDAHist LANGUAGES CXX CUDA)
# target_sources(CUDAHist src/RHnCUDA.cuh)

# message("INCKUDING "   ${CUDA_INCLUDE_DIRS})
# target_include_directories(CUDAHist PRIVATE ${CUDA_INCLUDE_DIRS})
# target_link_libraries(CUDAHist PRIVATE ${CUDA_CUBLAS_LIBRARIES})

ROOT_LINKER_LIBRARY(CUDAHist src/RHnCUDA.cu inc/CUDAHelpers.cuh TYPE SHARED DEPENDENCIES MathCore)
target_compile_options(CUDAHist PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
# target_include_directories(CUDAHist PRIVATE ${CUDA_INCLUDE_DIRS})
set_target_properties(CUDAHist PROPERTIES
  # LANGUAGE CUDA
  # LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
)

# set(Libraries ROOTDataFrame)
# include_directories(${CUDA_INCLUDE_DIRS})
# CUDA_ADD_EXECUTABLE(CUDAHist src/RHnCUDA.cxx)
# TARGET_LINK_LIBRARIES(ROOTDataFrame inc/RHnCUDA.cuh )



# #   CUDA_RUNTIME_LIBRARY Shared
ROOT_ADD_TEST_SUBDIRECTORY(test)
