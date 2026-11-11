# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA SOFIE package
############################################################################
#Author: Sitong An, Lorenzo Moneta 10/03/2021

add_library(ROOTTMVASofieParser SHARED)
target_sources(ROOTTMVASofieParser
PRIVATE
  src/RModelParser_ONNX.cxx
  src/ParseBasicUnary.cxx
  src/ParseBasicBinary.cxx
  src/ParseBasicIs.cxx
  src/ParseBatchNormalization.cxx
  src/ParseCast.cxx
  src/ParseConcat.cxx
  src/ParseConv.cxx
  src/ParseConvTranspose.cxx
  src/ParseGemm.cxx
  src/ParseGRU.cxx
  src/ParseIdentity.cxx
  src/ParseLeakyRelu.cxx
  src/ParseLSTM.cxx
  src/ParseBasicNary.cxx
  src/ParsePool.cxx
  src/ParseReduce.cxx
  src/ParseRelu.cxx
  src/ParseReshape.cxx
  src/ParseRNN.cxx
  src/ParseSelu.cxx
  src/ParseShape.cxx
  src/ParseSigmoid.cxx
  src/ParseSwish.cxx
  src/ParseSlice.cxx
  src/ParseSoftmax.cxx
  src/ParseTanh.cxx
  src/ParseTranspose.cxx
  src/ParseErf.cxx
  src/ParseRange.cxx
  src/ParseLayerNormalization.cxx
  src/ParseInstanceNormalization.cxx
  src/ParseExpand.cxx
  src/ParseGather.cxx
  src/ParseGatherND.cxx
  src/ParseElu.cxx
  src/ParseFuseConvAdd.cxx
  src/ParseFuseConvTransposeAdd.cxx
  src/ParseFuseGemmRelu.cxx
  src/ParseFuseBatchnormRelu.cxx
  src/ParseFuseMatMulAdd.cxx
  src/ParseMatMul.cxx
  src/ParseComparision.cxx
  src/ParseEyeLike.cxx
  src/ParseConstant.cxx
  src/ParseTopK.cxx
  src/ParseTile.cxx
  src/ParseSplit.cxx
  src/ParseIf.cxx
  src/ParsePad.cxx
  src/ParseWhere.cxx
  src/ParseEinsum.cxx
  src/ParseRandom.cxx
  src/ParseScatterElements.cxx
  src/ParseScatterND.cxx
  src/ParseNonZero.cxx
  src/ParseNot.cxx
  src/ParseClip.cxx
  src/ParseGelu.cxx
  src/ParseHardSigmoid.cxx
  src/ParseHardSwish.cxx
  src/ParseArgMax.cxx
  src/ParsePRelu.cxx
PRIVATE
  FILE_SET private_header_files TYPE HEADERS BASE_DIRS src/ FILES
    # The ONNX model files are read with a small self-contained protobuf
    # wire-format decoder (src/onnx.hxx), so this package has no external
    # dependency on protobuf / libprotobuf / protoc.
    src/onnx.hxx
PUBLIC
  FILE_SET HEADERS TYPE HEADERS BASE_DIRS inc/ FILES
    inc/TMVA/RModelParser_ONNX.hxx
  FILE_SET linkDef TYPE HEADERS BASE_DIRS inc/ FILES
    inc/LinkDef.h
)
target_link_libraries(ROOTTMVASofieParser PUBLIC ROOTTMVASofie PRIVATE Core)
ROOT_GENERATE_TARGET_DICTIONARY(ROOTTMVASofieParser "")
ROOT_INSTALL_TARGET(ROOTTMVASofieParser)

# workaround to include privately headers that are not public, from a "friend" target
get_target_property(TmvaSofie_Friend_Dirs ROOTTMVASofie HEADER_DIRS_friend_header_files)
target_include_directories(ROOTTMVASofieParser PRIVATE ${TmvaSofie_Friend_Dirs})
