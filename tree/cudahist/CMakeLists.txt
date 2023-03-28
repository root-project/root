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

ROOT_STANDARD_LIBRARY_PACKAGE(CUDAHist
  HEADERS
    RHnCUDA.h
  SOURCES
    RHnCUDA.cxx
  DEPENDENCIES
    CUDAHistLink
)

ROOT_LINKER_LIBRARY(CUDAHistLink  src/RHnCUDA.cu inc/CUDAHelpers.cuh TYPE SHARED DEPENDENCIES MathCore)
target_compile_options(CUDAHistLink  PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
set_target_properties(CUDAHistLink PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

ROOT_ADD_TEST_SUBDIRECTORY(test)
