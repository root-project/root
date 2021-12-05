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
   TMVA/ROperator_Add.hxx
   TMVA/ROperator_BatchNormalization.hxx
   TMVA/ROperator_Conv.hxx
   TMVA/ROperator_Gemm.hxx
   TMVA/ROperator_Relu.hxx
   TMVA/ROperator_Reshape.hxx
   TMVA/ROperator_Selu.hxx
   TMVA/ROperator_Sigmoid.hxx
   TMVA/ROperator_Slice.hxx
   TMVA/ROperator_Transpose.hxx
   TMVA/ROperator_Pool.hxx
   TMVA/ROperator_Concat.hxx
   TMVA/SOFIE_common.hxx
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
