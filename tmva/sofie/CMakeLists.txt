# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA SOFIE package
############################################################################
#Author: Sitong An, Lorenzo Moneta 10/03/2011

include_directories(SYSTEM ${Protobuf_INCLUDE_DIRS})

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "onnx.proto3")
set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)

ROOT_STANDARD_LIBRARY_PACKAGE(SOFIE
  HEADERS
    TMVA/OperatorList.hxx
    TMVA/RModel.hxx
    TMVA/RModelParser_ONNX.hxx
    TMVA/ROperator.hxx
    TMVA/ROperator_Gemm.hxx
    TMVA/ROperator_Relu.hxx
    TMVA/ROperator_Transpose.hxx
    TMVA/SOFIE_common.hxx
    ${PROTO_HDRS}
  SOURCES
    src/RModel.cxx
    src/RModelParser_ONNX.cxx
    src/SOFIE_common.cxx
    ${PROTO_SRCS}
  LIBRARIES
    ${Protobuf_LIBRARIES}
)

message(STATUS ${PROTO_SRCS})
message(STATUS ${PROTO_HDRS})
message(STATUS ${Protobuf_LIBRARIES})
