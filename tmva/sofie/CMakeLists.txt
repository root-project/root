# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA SOFIE package
############################################################################
#Author: Sitong An, Lorenzo Moneta 10/03/2021

add_library(ROOTTMVASofie SHARED)
target_sources(ROOTTMVASofie
PRIVATE
  src/RModel.cxx
  src/SOFIE_common.cxx
  src/SOFIE_common_helpers.cxx
PRIVATE
  # The ROperator*.hxx headers are an implementation detail of the model
  # generation and stay private to this library and ROOTTMVASofieParser, which
  # resolve them via private inclusion of the "friend" header fileset's include directory
  FILE_SET friend_header_files TYPE HEADERS BASE_DIRS inc/ FILES
    inc/TMVA/ROperator.hxx
    inc/TMVA/ROperator_ArgMax.hxx
    inc/TMVA/ROperator_BasicBinary.hxx
    inc/TMVA/ROperator_BasicNary.hxx
    inc/TMVA/ROperator_BasicUnary.hxx
    inc/TMVA/ROperator_Basic_Is.hxx
    inc/TMVA/ROperator_BatchNormalization.hxx
    inc/TMVA/ROperator_Cast.hxx
    inc/TMVA/ROperator_Clip.hxx
    inc/TMVA/ROperator_Comparision.hxx
    inc/TMVA/ROperator_Concat.hxx
    inc/TMVA/ROperator_Constant.hxx
    inc/TMVA/ROperator_Conv.hxx
    inc/TMVA/ROperator_ConvTranspose.hxx
    inc/TMVA/ROperator_Einsum.hxx
    inc/TMVA/ROperator_Elu.hxx
    inc/TMVA/ROperator_Erf.hxx
    inc/TMVA/ROperator_Expand.hxx
    inc/TMVA/ROperator_EyeLike.hxx
    inc/TMVA/ROperator_GRU.hxx
    inc/TMVA/ROperator_Gather.hxx
    inc/TMVA/ROperator_GatherND.hxx
    inc/TMVA/ROperator_Gelu.hxx
    inc/TMVA/ROperator_Gemm.hxx
    inc/TMVA/ROperator_HardSigmoid.hxx
    inc/TMVA/ROperator_HardSwish.hxx
    inc/TMVA/ROperator_Identity.hxx
    inc/TMVA/ROperator_InstanceNormalization.hxx
    inc/TMVA/ROperator_LSTM.hxx
    inc/TMVA/ROperator_LayerNormalization.hxx
    inc/TMVA/ROperator_LeakyRelu.hxx
    inc/TMVA/ROperator_NonZero.hxx
    inc/TMVA/ROperator_Not.hxx
    inc/TMVA/ROperator_PRelu.hxx
    inc/TMVA/ROperator_Pad.hxx
    inc/TMVA/ROperator_Pool.hxx
    inc/TMVA/ROperator_RNN.hxx
    inc/TMVA/ROperator_Random.hxx
    inc/TMVA/ROperator_Range.hxx
    inc/TMVA/ROperator_Reduce.hxx
    inc/TMVA/ROperator_Relu.hxx
    inc/TMVA/ROperator_Reshape.hxx
    inc/TMVA/ROperator_ScatterElements.hxx
    inc/TMVA/ROperator_ScatterND.hxx
    inc/TMVA/ROperator_Selu.hxx
    inc/TMVA/ROperator_Shape.hxx
    inc/TMVA/ROperator_Sigmoid.hxx
    inc/TMVA/ROperator_Slice.hxx
    inc/TMVA/ROperator_Softmax.hxx
    inc/TMVA/ROperator_Split.hxx
    inc/TMVA/ROperator_SubGraph.hxx
    inc/TMVA/ROperator_Swish.hxx
    inc/TMVA/ROperator_Tanh.hxx
    inc/TMVA/ROperator_Tile.hxx
    inc/TMVA/ROperator_TopK.hxx
    inc/TMVA/ROperator_Transpose.hxx
    inc/TMVA/ROperator_Where.hxx
PUBLIC
  FILE_SET HEADERS TYPE HEADERS BASE_DIRS inc/ FILES
    inc/TMVA/RModel.hxx
    inc/TMVA/SOFIE_common.hxx
  FILE_SET linkDef TYPE HEADERS BASE_DIRS inc/ FILES
    inc/LinkDef.h
)
target_link_libraries(ROOTTMVASofie PUBLIC Core)

ROOT_GENERATE_TARGET_DICTIONARY(ROOTTMVASofie "")
ROOT_INSTALL_TARGET(ROOTTMVASofie)

# Tests are not enabled unconditionally because they require BLAS
if(test_tmva_sofie)
  ROOT_ADD_TEST_SUBDIRECTORY(test)
endif()
