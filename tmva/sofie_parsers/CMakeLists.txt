# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA SOFIE package
############################################################################
#Author: Sitong An, Lorenzo Moneta 10/03/2021

# The ONNX model files are read with a small self-contained protobuf
# wire-format decoder (src/onnx.hxx), so this package has no external
# dependency on protobuf / libprotobuf / protoc.

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTTMVASofieParser
  HEADERS
   TMVA/RModelParser_ONNX.hxx
  SOURCES
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
  DEPENDENCIES
    TMVA
    ROOTTMVASofie
)

set_target_properties(ROOTTMVASofieParser PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE)
