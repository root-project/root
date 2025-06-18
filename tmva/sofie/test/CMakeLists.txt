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

#Finding .onnx files to be parsed and creating the appropriate code to
# parse all file. It is much faster to combine all parsing in a single executable
# which will avoid initialization time (especially when using ROOT)
set(CAPTURE_STR "EmitModel( \"@1\", \"@2\");")
set(ALL_CAPTURES "")
# Finding .onnx files to be parsed and creating the appropriate command
file(GLOB ONNX_FILES "${ONNX_MODELS_DIR}/*.onnx")
foreach(onnx_file ${ONNX_FILES})
	get_filename_component(fname ${onnx_file} NAME_WE)
	get_filename_component(fdir ${onnx_file} DIRECTORY)
  string(REPLACE "@1" ${onnx_file} cap ${CAPTURE_STR})
  string(REPLACE "@2" ${fname} cap ${cap})
  list(APPEND ALL_CAPTURES ${cap})
endforeach()
string(REPLACE ";" ";\n" EMIT_CAPTURES "${ALL_CAPTURES}")
configure_file(EmitFromONNX.cxx.in EmitFromONNX_all.cxx @ONLY)
configure_file(EmitFromRoot.cxx.in EmitFromRoot_all.cxx @ONLY)

add_executable(emitFromONNX
   EmitFromONNX_all.cxx
)
target_include_directories(emitFromONNX PRIVATE
   ${CMAKE_SOURCE_DIR}/tmva/sofie/inc
   ${SOFIE_PARSERS_DIR}/inc
   ${CMAKE_SOURCE_DIR}/tmva/inc
   ${CMAKE_CURRENT_BINARY_DIR}   # this is for the protobuf headerfile
)

target_link_libraries(emitFromONNX protobuf::libprotobuf ROOTTMVASofie ROOTTMVASofieParser)
set_target_properties(emitFromONNX PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
## silence protobuf warnings seen in version 3.0 and 3.6. Not needed from protobuf version 3.17
target_compile_options(emitFromONNX PRIVATE -Wno-unused-parameter -Wno-array-bounds)

# Automatic compilation of headers from onnx files
add_custom_target(SofieCompileModels_ONNX)
add_dependencies(SofieCompileModels_ONNX emitFromONNX)


add_custom_command(TARGET SofieCompileModels_ONNX POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1 ./emitFromONNX ${onnx_file} ${CMAKE_CURRENT_BINARY_DIR}/${fname}
		USES_TERMINAL )


# Creating a Google Test
if (BLAS_FOUND)  # we need BLAS for compiling the models
ROOT_ADD_GTEST(TestCustomModelsFromONNX TestCustomModelsFromONNX.cxx
  LIBRARIES
    MathCore
    ROOTTMVASofie
    BLAS::BLAS
  INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_dependencies(TestCustomModelsFromONNX SofieCompileModels_ONNX)
endif()

#For testing serialisation of RModel object

add_executable(emitFromROOT
   EmitFromRoot_all.cxx
)
target_include_directories(emitFromROOT PRIVATE
   ${CMAKE_SOURCE_DIR}/tmva/sofie/inc
   ${SOFIE_PARSERS_DIR}/inc
   ${CMAKE_SOURCE_DIR}/tmva/inc
   ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(emitFromROOT protobuf::libprotobuf RIO ROOTTMVASofie ROOTTMVASofieParser)
set_target_properties(emitFromROOT PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
## silence protobuf warnings seen in version 3.0 and 3.6. Not needed from protobuf version 3.17
target_compile_options(emitFromROOT PRIVATE -Wno-unused-parameter -Wno-array-bounds)

# Automatic compilation of headers from root files
add_custom_target(SofieCompileModels_ROOT)
# onepcm or modules dependency is needed for using ROOT I/O when converting a model in a ROOT file
if (runtime_cxxmodules)
  add_dependencies(SofieCompileModels_ROOT emitFromROOT modules_idx)
else()
  add_dependencies(SofieCompileModels_ROOT emitFromROOT onepcm)
endif()

add_custom_command(TARGET SofieCompileModels_ROOT POST_BUILD
COMMAND ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1 ./emitFromROOT
		USES_TERMINAL )

if (BLAS_FOUND)
  # Creating a Google Test for Serialisation of RModel
  ROOT_ADD_GTEST(TestCustomModelsFromROOT TestCustomModelsFromROOT.cxx
    LIBRARIES
      ROOTTMVASofie
      BLAS::BLAS
    INCLUDE_DIRS
      ${CMAKE_CURRENT_BINARY_DIR}
  )
  add_dependencies(TestCustomModelsFromROOT SofieCompileModels_ROOT)

  if (clad)
    # Creating a Google Test for the automatic differentiation of Gemm_Call
    ROOT_ADD_GTEST(TestGemmDerivative TestGemmDerivative.cxx
      LIBRARIES
        Core
        BLAS::BLAS
      INCLUDE_DIRS
        ${CMAKE_CURRENT_BINARY_DIR}
    )
  endif()
endif()

# gtest
# Look for needed python modules
ROOT_FIND_PYTHON_MODULE(torch)
if (ROOT_TORCH_FOUND)
  configure_file(Conv1dModelGenerator.py  Conv1dModelGenerator.py COPYONLY)
  configure_file(Conv2dModelGenerator.py  Conv2dModelGenerator.py COPYONLY)
  configure_file(Conv3dModelGenerator.py  Conv3dModelGenerator.py COPYONLY)
  configure_file(ConvTrans2dModelGenerator.py  ConvTrans2dModelGenerator.py COPYONLY)
  configure_file(LinearModelGenerator.py  LinearModelGenerator.py COPYONLY)
  configure_file(RecurrentModelGenerator.py  RecurrentModelGenerator.py COPYONLY)

  if (BLAS_FOUND)
    ROOT_ADD_GTEST(TestSofieModels TestSofieModels.cxx
      LIBRARIES
        ROOTTMVASofie
        ROOTTMVASofieParser
        BLAS::BLAS
      INCLUDE_DIRS
        ${CMAKE_CURRENT_BINARY_DIR}
    )
  endif()
endif()

add_executable(emitGNN
              GNN/EmitGNN.cxx
              ${CMAKE_SOURCE_DIR}/tmva/sofie/src/RModel_GNN.cxx
              ${CMAKE_SOURCE_DIR}/tmva/sofie/src/SOFIE_common.cxx
)
target_include_directories(emitGNN PRIVATE
   ${CMAKE_SOURCE_DIR}/tmva/sofie/inc
   ${CMAKE_SOURCE_DIR}/tmva/inc
   ${CMAKE_SOURCE_DIR}/core/foundation/inc
   ${CMAKE_BINARY_DIR}/ginclude   # this is for RConfigure.h
)
target_link_libraries(emitGNN ROOTTMVASofie)
set_target_properties(emitGNN PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
## silence protobuf warnings seen in version 3.0 and 3.6. Not needed from protobuf version 3.17
target_compile_options(emitGNN PRIVATE -Wno-unused-parameter -Wno-array-bounds)

add_executable(emitGraphIndependent
              GNN/EmitGraphIndependent.cxx
              ${CMAKE_SOURCE_DIR}/tmva/sofie/src/RModel_GraphIndependent.cxx
              ${CMAKE_SOURCE_DIR}/tmva/sofie/src/SOFIE_common.cxx
)
target_include_directories(emitGraphIndependent PRIVATE
   ${CMAKE_SOURCE_DIR}/tmva/sofie/inc
   ${CMAKE_SOURCE_DIR}/tmva/inc
   ${CMAKE_SOURCE_DIR}/core/foundation/inc
   ${CMAKE_BINARY_DIR}/ginclude   # this is for RConfigure.h
)
target_link_libraries(emitGraphIndependent ROOTTMVASofie)
set_target_properties(emitGraphIndependent PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
## silence protobuf warnings seen in version 3.0 and 3.6. Not needed from protobuf version 3.17
target_compile_options(emitGraphIndependent PRIVATE -Wno-unused-parameter -Wno-array-bounds)
