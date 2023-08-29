# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA SOFIE package
############################################################################
#Author: Sitong An, Lorenzo Moneta 10/03/2021

#sofie is built only if protobuf is found

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTTMVASofie
  HEADERS
   TMVA/OperatorList.hxx
   TMVA/RModel.hxx
   TMVA/ROperator.hxx
   TMVA/ROperator_BasicUnary.hxx
   TMVA/ROperator_BasicBinary.hxx
   TMVA/ROperator_BasicNary.hxx
   TMVA/ROperator_BatchNormalization.hxx
   TMVA/ROperator_Conv.hxx
   TMVA/ROperator_ConvTranspose.hxx
   TMVA/ROperator_Gemm.hxx
   TMVA/ROperator_Relu.hxx
   TMVA/ROperator_Tanh.hxx
   TMVA/ROperator_LeakyRelu.hxx
   TMVA/ROperator_Reshape.hxx
   TMVA/ROperator_Selu.hxx
   TMVA/ROperator_Sigmoid.hxx
   TMVA/ROperator_Slice.hxx
   TMVA/ROperator_Transpose.hxx
   TMVA/ROperator_Pool.hxx
   TMVA/ROperator_Concat.hxx
   TMVA/ROperator_Identity.hxx
   TMVA/ROperator_Softmax.hxx
   TMVA/ROperator_Reduce.hxx
   TMVA/ROperator_Shape.hxx
   TMVA/ROperator_Cast.hxx
   TMVA/ROperator_Custom.hxx
   TMVA/ROperator_LayerNormalization.hxx
   TMVA/ROperator_Expand.hxx
   TMVA/ROperator_Erf.hxx
   TMVA/SOFIE_common.hxx
   TMVA/SOFIEHelpers.hxx
  SOURCES
    src/RModel.cxx
    src/SOFIE_common.cxx
  DEPENDENCIES
    TMVA
)

target_include_directories(ROOTTMVASofie PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

if (NOT TARGET_GPU)
  set(TARGET_GPU Intel)
endif()

if (NOT GPU_BLAS)
  set(GPU_BLAS MKLBLAS)
endif()

if (tmva-sofie AND sofie-sycl)
  set(TARGET_GPUS "Intel;NVIDIA;AMD")
  set(BLAS_GPU_IMPLEMENTATIONS "MKLBLAS;portBLAS")
  set(MKL_TARGETS "Intel")

  if (NOT ${TARGET_GPU} IN_LIST TARGET_GPUS)
    message(FATAL_ERROR "Please select a GPU target with -DTARGET_GPU. Valid values are: [Intel, NVIDIA, AMD]")
  endif()

  if (NOT ${GPU_BLAS} IN_LIST BLAS_GPU_IMPLEMENTATIONS)
    message(FATAL_ERROR "Please select a BLAS Implementation with -DGPU_BLAS. Valid values are: [MKLBLAS, portBLAS]")
  endif()
  
  if (NOT ${TARGET_GPU} IN_LIST MKL_TARGETS AND ${GPU_BLAS} STREQUAL "MKLBLAS")
    message(FATAL_ERROR "Target GPU is not compatible with MKLBLAS. Please select portBLAS instead")
  endif()

endif()
set_target_properties(ROOTTMVASofie PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE)

target_compile_definitions(ROOTTMVASofie PUBLIC TARGET_GPU=${TARGET_GPU})
target_compile_definitions(ROOTTMVASofie PUBLIC GPU_BLAS=${GPU_BLAS})

# tests requires protobuf
if (tmva-sofie)
  ROOT_ADD_TEST_SUBDIRECTORY(test)
endif()

