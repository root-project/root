# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA SOFIE tests.
# @author Federico Sossai, Sanjiban Sengupta
############################################################################


if (NOT ONNX_MODELS_DIR)
  set(ONNX_MODELS_DIR input_models)
endif()

# Finding .onnx files to be parsed and creating the appropriate code to
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

ROOTTEST_GENERATE_EXECUTABLE(emitFromONNX EmitFromONNX_all.cxx
                             LIBRARIES protobuf::libprotobuf ROOTTMVASofie ROOTTMVASofieParser
                             FIXTURES_SETUP sofie-compile-models-onnx-build)

# silence protobuf warnings seen in version 3.0 and 3.6. Not needed from protobuf version 3.17
target_compile_options(emitFromONNX PRIVATE -Wno-unused-parameter -Wno-array-bounds)

ROOTTEST_ADD_TEST(SofieCompileModels_ONNX
  COMMAND ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1 ./emitFromONNX ${onnx_file} ${CMAKE_CURRENT_BINARY_DIR}/${fname}
  FIXTURES_REQUIRED sofie-compile-models-onnx-build
  FIXTURES_SETUP sofie-compile-models-onnx
)

# Creating a Google Test
if (BLAS_FOUND)  # we need BLAS for compiling the models
  ROOTTEST_GENERATE_EXECUTABLE(TestCustomModelsFromONNX TestCustomModelsFromONNX.cxx
    LIBRARIES
      MathCore
      ROOTTMVASofie
      BLAS::BLAS
      GTest::gtest
      GTest::gtest_main
    FIXTURES_REQUIRED
      sofie-compile-models-onnx
    FIXTURES_SETUP
      sofie-test-models-onnx-build
  )
  target_include_directories(TestCustomModelsFromONNX PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  ROOTTEST_ADD_TEST(TestCustomModelsFromONNX
                    EXEC ./TestCustomModelsFromONNX
                    FIXTURES_REQUIRED sofie-test-models-onnx-build)
endif()

# For testing serialisation of RModel object

ROOTTEST_GENERATE_EXECUTABLE(emitFromROOT EmitFromRoot_all.cxx
    LIBRARIES protobuf::libprotobuf RIO ROOTTMVASofie ROOTTMVASofieParser
    FIXTURES_SETUP sofie-compile-models-onnx-root
)
# silence protobuf warnings seen in version 3.0 and 3.6. Not needed from protobuf version 3.17
target_compile_options(emitFromROOT PRIVATE -Wno-unused-parameter -Wno-array-bounds)

# Automatic compilation of headers from root files
ROOTTEST_ADD_TEST(SofieCompileModels_ROOT
  COMMAND ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1 ./emitFromROOT
  FIXTURES_REQUIRED sofie-compile-models-onnx-root
  FIXTURES_SETUP sofie-compile-models-root
)

if (BLAS_FOUND)
  # Creating a Google Test for Serialisation of RModel
  ROOTTEST_GENERATE_EXECUTABLE(TestCustomModelsFromROOT TestCustomModelsFromROOT.cxx
    LIBRARIES
      ROOTTMVASofie
      BLAS::BLAS
      GTest::gtest
      GTest::gtest_main
    FIXTURES_REQUIRED
      sofie-compile-models-root
    FIXTURES_SETUP
      sofie-test-models-root-build
  )
  target_include_directories(TestCustomModelsFromROOT PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  ROOTTEST_ADD_TEST(TestCustomModelsFromROOT
                    EXEC ./TestCustomModelsFromROOT
                    FIXTURES_REQUIRED sofie-test-models-root-build)

  if (clad)
    # Creating a Google Test for the automatic differentiation of Gemm_Call
    ROOT_ADD_GTEST(TestGemmDerivative TestGemmDerivative.cxx LIBRARIES Core BLAS::BLAS)
  endif()
endif()

# Look for needed Python modules
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

ROOT_EXECUTABLE(emitGNN GNN/EmitGNN.cxx LIBRARIES ROOTTMVASofie)
ROOT_ADD_TEST(tmva-sofie-EmitGNN COMMAND emitGNN)

ROOT_EXECUTABLE(EmitGraphIndependent GNN/EmitGraphIndependent.cxx LIBRARIES ROOTTMVASofie)
ROOT_ADD_TEST(tmva-sofie-EmitGraphIndependent COMMAND EmitGraphIndependent)
