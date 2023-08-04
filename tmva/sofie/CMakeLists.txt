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
set_target_properties(ROOTTMVASofie PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE)

# tests requires protobuf
if (tmva-sofie)
  ROOT_ADD_TEST_SUBDIRECTORY(test)
endif()

if (tmva-sycl)
  find_package(IntelSYCL REQUIRED)
  find_package(MKL REQUIRED)
  target_compile_options(ROOTTMVASofie PUBLIC -fsycl -DMKL_ILP64  -I"${MKLROOT}/include")
  target_link_options(ROOTTMVASofie PUBLIC -fsycl  -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lpthread -lm -ldl)
  add_sycl_to_target(TARGET ROOTTMVASofie SOURCES src/SOFIE_common.cxx src RModel.cxx)
endif()
