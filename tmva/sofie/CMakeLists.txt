# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA SOFIE package
############################################################################
#Author: Sitong An, Lorenzo Moneta 10/03/2021

# Only the headers listed below are copied to the build tree and installed.
set(TMVASOFIE_HEADERS
   TMVA/RModel.hxx
   TMVA/SOFIE_common.hxx
)
# The ROperator*.hxx headers are an implementation detail of the model
# generation and stay private to this library and ROOTTMVASofieParser, which
# resolve them via the public build-interface include directory of this
# target.
set(TMVASOFIE_EXTRA_HEADERS
  TMVA/ROperator.hxx
  TMVA/ROperator_ArgMax.hxx
  TMVA/ROperator_BasicBinary.hxx
  TMVA/ROperator_BasicNary.hxx
  TMVA/ROperator_BasicUnary.hxx
  TMVA/ROperator_Basic_Is.hxx
  TMVA/ROperator_BatchNormalization.hxx
  TMVA/ROperator_Cast.hxx
  TMVA/ROperator_Clip.hxx
  TMVA/ROperator_Comparision.hxx
  TMVA/ROperator_Concat.hxx
  TMVA/ROperator_Constant.hxx
  TMVA/ROperator_Conv.hxx
  TMVA/ROperator_ConvTranspose.hxx
  TMVA/ROperator_Einsum.hxx
  TMVA/ROperator_Elu.hxx
  TMVA/ROperator_Erf.hxx
  TMVA/ROperator_Expand.hxx
  TMVA/ROperator_EyeLike.hxx
  TMVA/ROperator_GRU.hxx
  TMVA/ROperator_Gather.hxx
  TMVA/ROperator_GatherND.hxx
  TMVA/ROperator_Gelu.hxx
  TMVA/ROperator_Gemm.hxx
  TMVA/ROperator_HardSigmoid.hxx
  TMVA/ROperator_HardSwish.hxx
  TMVA/ROperator_Identity.hxx
  TMVA/ROperator_InstanceNormalization.hxx
  TMVA/ROperator_LSTM.hxx
  TMVA/ROperator_LayerNormalization.hxx
  TMVA/ROperator_LeakyRelu.hxx
  TMVA/ROperator_NonZero.hxx
  TMVA/ROperator_Not.hxx
  TMVA/ROperator_PRelu.hxx
  TMVA/ROperator_Pad.hxx
  TMVA/ROperator_Pool.hxx
  TMVA/ROperator_RNN.hxx
  TMVA/ROperator_Random.hxx
  TMVA/ROperator_Range.hxx
  TMVA/ROperator_Reduce.hxx
  TMVA/ROperator_Relu.hxx
  TMVA/ROperator_Reshape.hxx
  TMVA/ROperator_ScatterElements.hxx
  TMVA/ROperator_ScatterND.hxx
  TMVA/ROperator_Selu.hxx
  TMVA/ROperator_Shape.hxx
  TMVA/ROperator_Sigmoid.hxx
  TMVA/ROperator_Slice.hxx
  TMVA/ROperator_Softmax.hxx
  TMVA/ROperator_Split.hxx
  TMVA/ROperator_SubGraph.hxx
  TMVA/ROperator_Swish.hxx
  TMVA/ROperator_Tanh.hxx
  TMVA/ROperator_Tile.hxx
  TMVA/ROperator_TopK.hxx
  TMVA/ROperator_Transpose.hxx
  TMVA/ROperator_Where.hxx
)
  
ROOT_STANDARD_LIBRARY_PACKAGE(ROOTTMVASofie
  
  NO_GLOB_HEADERS
  HEADERS
    ${TMVASOFIE_HEADERS}
  SOURCES
    src/RModel.cxx
    src/SOFIE_common.cxx
    src/SOFIE_common_helpers.cxx
  DEPENDENCIES
    Core
)

set(RELATIVE_INC_HEADERS ${TMVASOFIE_HEADERS} ${TMVASOFIE_EXTRA_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(ROOTTMVASofie
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

# Tests are not enabled unconditionally because they require BLAS
if(test_tmva_sofie)
  ROOT_ADD_TEST_SUBDIRECTORY(test)
endif()
