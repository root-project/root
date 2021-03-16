# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA SOFIE package
############################################################################
#Author: Sitong An, Lorenzo Moneta 10/03/2011

include_directories(SYSTEM ${Protobuf_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} )

#add_custom_target(proto_dep DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "onnx_proto3")
set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)
#set_source_files_properties(onnx.proto3.pb.cc onnx.proto3.pb.h PROPERTIES GENERATED TRUE)

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
   #onnx.proto3.pb.h
  SOURCES
    src/RModel.cxx
    src/RModelParser_ONNX.cxx
    src/SOFIE_common.cxx
    #onnx.proto3.pb.cc
    ${PROTO_SRCS}

  LIBRARIES
    ${Protobuf_LIBRARIES}
  DEPENDENCIES
    TMVA
   NO_CXXMODULE
)

message(STATUS ${PROTO_SRCS})
message(STATUS ${PROTO_HDRS})
message(STATUS ${Protobuf_LIBRARIES})
