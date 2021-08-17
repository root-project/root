# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA SOFIE tests.
# @author Federico Sossai, Sanjiban Sengupta
############################################################################


set(SOFIE_PARSERS_DIR ${CMAKE_SOURCE_DIR}/tmva/sofie_parsers)

if (NOT ONNX_MODELS_DIR)
  set(ONNX_MODELS_DIR input_models)
endif()

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "${SOFIE_PARSERS_DIR}/onnx_proto3")
set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)

add_executable(emitFromONNX
   EmitFromONNX.cxx
   ${CMAKE_SOURCE_DIR}/tmva/sofie/src/SOFIE_common.cxx
   ${CMAKE_SOURCE_DIR}/tmva/sofie/src/RModel.cxx
   ${SOFIE_PARSERS_DIR}/src/RModelParser_ONNX.cxx
   ${PROTO_SRCS}
)
target_include_directories(emitFromONNX PRIVATE
   ${CMAKE_SOURCE_DIR}/tmva/sofie/inc
   ${SOFIE_PARSERS_DIR}/inc
   ${CMAKE_SOURCE_DIR}/tmva/inc
   ${Protobuf_INCLUDE_DIRS}
   ${CMAKE_CURRENT_BINARY_DIR}   # this is for the protobuf headerfile
   ${CMAKE_SOURCE_DIR}/core/foundation/inc
   ${CMAKE_BINARY_DIR}/ginclude   # this is for RConfigure.h
)

target_link_libraries(emitFromONNX ${Protobuf_LIBRARIES} ROOTTMVASofie)
set_target_properties(emitFromONNX PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE)

# Automatic compilation of headers from onnx files
add_custom_target(SofieCompileModels_ONNX)
add_dependencies(SofieCompileModels_ONNX emitFromONNX)

# Finding .onnx files to be compiled and creating the appropriate command
file(GLOB ONNX_FILES "${ONNX_MODELS_DIR}/*.onnx")
foreach(onnx_file ${ONNX_FILES})
	get_filename_component(fname ${onnx_file} NAME_WE)
	get_filename_component(fdir ${onnx_file} DIRECTORY)
	add_custom_command(TARGET SofieCompileModels_ONNX POST_BUILD
		COMMAND ./emitFromONNX ${onnx_file} ${CMAKE_CURRENT_BINARY_DIR}/${fname}
		USES_TERMINAL
	)
endforeach()

# Creating a Google Test
ROOT_ADD_GTEST(TestCustomModelsFromONNX TestCustomModelsFromONNX.cxx
  LIBRARIES
    ROOTTMVASofie
    blas
  INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_dependencies(TestCustomModelsFromONNX SofieCompileModels_ONNX)


#For testing serialisation of RModel object
add_executable(emitFromROOT
   EmitFromRoot.cxx
   ${CMAKE_SOURCE_DIR}/tmva/sofie/src/SOFIE_common.cxx
   ${CMAKE_SOURCE_DIR}/tmva/sofie/src/RModel.cxx
   ${SOFIE_PARSERS_DIR}/src/RModelParser_ONNX.cxx
   ${PROTO_SRCS}
)
target_include_directories(emitFromROOT PRIVATE
   ${CMAKE_SOURCE_DIR}/tmva/sofie/inc
   ${SOFIE_PARSERS_DIR}/inc
   ${CMAKE_SOURCE_DIR}/tmva/inc
   ${Protobuf_INCLUDE_DIRS}
   ${CMAKE_CURRENT_BINARY_DIR}   # this is for the protobuf headerfile
   ${CMAKE_SOURCE_DIR}/core/foundation/inc
   ${CMAKE_BINARY_DIR}/ginclude   # this is for RConfigure.h
)
target_link_libraries(emitFromROOT ${Protobuf_LIBRARIES} ROOTTMVASofie)
set_target_properties(emitFromROOT PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE)

# Automatic compilation of headers from root files
add_custom_target(SofieCompileModels_ROOT)
add_dependencies(SofieCompileModels_ROOT emitFromROOT)

# Finding .onnx files to be parsed and creating the appropriate command
file(GLOB ONNX_FILES "${ONNX_MODELS_DIR}/*.onnx")
foreach(onnx_file ${ONNX_FILES})
	get_filename_component(fname ${onnx_file} NAME_WE)
	get_filename_component(fdir ${onnx_file} DIRECTORY)
	add_custom_command(TARGET SofieCompileModels_ROOT POST_BUILD
		COMMAND ./emitFromROOT ${onnx_file} ${CMAKE_CURRENT_BINARY_DIR}/${fname}
		USES_TERMINAL
	)
endforeach()

# Creating a Google Test for Serialisation of RModel
ROOT_ADD_GTEST(TestCustomModelsFromROOT TestCustomModelsFromROOT.cxx
  LIBRARIES
    ROOTTMVASofie
    blas
  INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(TestCustomModelsFromROOT SofieCompileModels_ROOT)
