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

ROOT_LINKER_LIBRARY(CUDAHist  src/RHnCUDA.cu inc/CUDAHelpers.cuh TYPE SHARED DEPENDENCIES MathCore)
target_compile_options(CUDAHist  PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
set_target_properties(CUDAHist PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

ROOT_ADD_TEST_SUBDIRECTORY(test)
