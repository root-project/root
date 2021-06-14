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

<<<<<<< HEAD
<<<<<<< HEAD
=======
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "onnx_proto3")
set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS_Development_Main} ${NUMPY_INCLUDE_DIRS})
=======

>>>>>>> 0ee3b32d3a... SOFIE restructured, RModelParser_Common added

>>>>>>> 83a03f946c... [TMVA][ROOT Storage] Initial Commit for PyKeras Converter
ROOT_STANDARD_LIBRARY_PACKAGE(ROOTTMVASofie
  HEADERS
   TMVA/OperatorList.hxx
   TMVA/RModel.hxx
<<<<<<< HEAD
<<<<<<< HEAD
=======
   TMVA/RModelParser_ONNX.hxx
   TMVA/RModelParser_Keras.hxx
>>>>>>> 83a03f946c... [TMVA][ROOT Storage] Initial Commit for PyKeras Converter
=======
>>>>>>> 0ee3b32d3a... SOFIE restructured, RModelParser_Common added
   TMVA/ROperator.hxx
   TMVA/ROperator_Gemm.hxx
   TMVA/ROperator_Relu.hxx
   TMVA/ROperator_Transpose.hxx
   TMVA/SOFIE_common.hxx
  SOURCES
    src/RModel.cxx
<<<<<<< HEAD
<<<<<<< HEAD
    src/SOFIE_common.cxx
=======
    src/RModelParser_ONNX.cxx
    src/RModelParser_Keras.cxx
=======
>>>>>>> 0ee3b32d3a... SOFIE restructured, RModelParser_Common added
    src/SOFIE_common.cxx
  LIBRARIES
    ${Protobuf_LIBRARIES}
<<<<<<< HEAD
    ${PYTHON_LIBRARIES_Development_Main}
>>>>>>> 83a03f946c... [TMVA][ROOT Storage] Initial Commit for PyKeras Converter
=======

>>>>>>> 0ee3b32d3a... SOFIE restructured, RModelParser_Common added
  DEPENDENCIES
    TMVA
)

target_include_directories(ROOTTMVASofie PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
set_target_properties(ROOTTMVASofie PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE)

ROOT_ADD_TEST_SUBDIRECTORY(test)