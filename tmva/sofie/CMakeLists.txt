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
   TMVA/RModel_Base.hxx
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
   TMVA/ROperator_Constant.hxx
   TMVA/ROperator_Softmax.hxx
   TMVA/ROperator_Reduce.hxx
   TMVA/ROperator_Shape.hxx
   TMVA/ROperator_Cast.hxx
   TMVA/ROperator_Custom.hxx
   TMVA/ROperator_LayerNormalization.hxx
   TMVA/ROperator_Expand.hxx
   TMVA/ROperator_Erf.hxx
   TMVA/ROperator_Swish.hxx
   TMVA/ROperator_Elu.hxx
   TMVA/ROperator_Comparision.hxx
   TMVA/ROperator_EyeLike.hxx
   TMVA/ROperator_Range.hxx
   TMVA/ROperator_TopK.hxx
   TMVA/ROperator_Tile.hxx
   TMVA/ROperator_Split.hxx
   TMVA/ROperator_SubGraph.hxx
   TMVA/ROperator_Pad.hxx
   TMVA/ROperator_Where.hxx
   TMVA/ROperator_Einsum.hxx
   TMVA/ROperator_Random.hxx
   TMVA/SOFIE_common.hxx
   TMVA/SOFIEHelpers.hxx

   TMVA/FunctionList.hxx
   TMVA/RModel_GNN.hxx
   TMVA/RModel_GraphIndependent.hxx
   TMVA/RFunction.hxx
   TMVA/RFunction_MLP.hxx
   TMVA/RFunction_Sum.hxx
   TMVA/RFunction_Mean.hxx
  SOURCES
    src/RModel_Base.cxx
    src/RModel.cxx
    src/RModel_GNN.cxx
    src/RModel_GraphIndependent.cxx
    src/RFunction.cxx
    src/RFunction_MLP.cxx
    src/RFunction_Mean.cxx
    src/RFunction_Sum.cxx
    src/SOFIE_common.cxx
  DEPENDENCIES
    TMVA
)

target_include_directories(ROOTTMVASofie PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

# default values for TARGET_GPU and GPU_BLAS in case the user does not specify them
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
    set(GPU_BLAS "portBLAS")
    message(WARNING "Target GPU is not compatible with MKLBLAS. Switching to portBLAS instead")
  endif()
endif()

set_target_properties(ROOTTMVASofie PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE)

if (${TARGET_GPU} STREQUAL "Intel")
  target_compile_definitions(ROOTTMVASofie PUBLIC TARGET_GPU=0)
elseif(${TARGET_GPU} STREQUAL "NVIDIA")
  target_compile_definitions(ROOTTMVASofie PUBLIC TARGET_GPU=1)
else()
  target_compile_definitions(ROOTTMVASofie PUBLIC TARGET_GPU=2)
endif()

if (${GPU_BLAS} STREQUAL "MKLBLAS")
  target_compile_definitions(ROOTTMVASofie PUBLIC GPU_BLAS=0)
else()
  target_compile_definitions(ROOTTMVASofie PUBLIC GPU_BLAS=1)
endif()

# tests requires protobuf
if (tmva-sofie)
   ROOT_ADD_TEST_SUBDIRECTORY(test)
endif()
