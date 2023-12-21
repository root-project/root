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

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "onnx_proto3")
set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)
## silence protobuf warnings for version 3.6. Not needed from protobuf version 3.17
set_source_files_properties(${PROTO_SRCS} PROPERTIES COMPILE_FLAGS -Wno-array-bounds)
## silence protobuf warnings for version 3.0. Not needed from protobuf version 3.6
set_source_files_properties(${PROTO_SRCS} PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTTMVASofieParser
  HEADERS
   TMVA/RModelParser_ONNX.hxx
  SOURCES
    src/RModelParser_ONNX.cxx
    src/ParseBasicUnary.cxx
    src/ParseBasicBinary.cxx
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
    src/ParseSlice.cxx
    src/ParseSoftmax.cxx
    src/ParseTanh.cxx
    src/ParseTranspose.cxx
    src/ParseErf.cxx
    src/ParseLayerNormalization.cxx
    src/ParseExpand.cxx
    src/ParseGather.cxx
    src/ParseElu.cxx
    src/ParseFuseConvAdd.cxx
    src/ParseFuseConvTransposeAdd.cxx
    src/ParseFuseMatMulAdd.cxx
    src/ParseMatMul.cxx
    src/ParseComparision.cxx
    src/ParseEyeLike.cxx
    ${PROTO_SRCS}
  LIBRARIES PUBLIC
    protobuf::libprotobuf
  DEPENDENCIES
    TMVA
    ROOTTMVASofie
)

target_include_directories(ROOTTMVASofieParser PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
set_target_properties(ROOTTMVASofieParser PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE)
