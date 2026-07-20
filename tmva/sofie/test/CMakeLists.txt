# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA SOFIE tests.
# @author Federico Sossai, Sanjiban Sengupta
############################################################################


# Look for needed Python modules
ROOT_FIND_PYTHON_MODULE(torch)

# onnx 1.19.0 has a bug that makes this version unusable:
# https://github.com/onnx/onnx/issues/7249
# 1.19.1 instead could be used
# In that case, we have to disable the "TestSofieModels" test,
# which imports onnx indirectly via torch.onnx
ROOT_FIND_PYTHON_MODULE(onnx)
if (ROOT_ONNX_FOUND AND DEFINED ROOT_ONNX_VERSION)
  if(ROOT_ONNX_VERSION VERSION_EQUAL "1.19.0")
    message(WARNING "Found broken onnx version ${ROOT_ONNX_VERSION} (see https://github.com/onnx/onnx/issues/7249). Some TMVA SOFIE tests will be disabled.")
    set(broken_onnx TRUE)
  endif()
endif()

# The ONNX input models are not checked into the repository: they are created
# by generate_input_models.py, which runs as the SofieGenerateModels_ONNX test
# that the other ONNX tests depend on. Set ONNX_MODELS_DIR to use pre-existing
# models from a custom directory instead.
if (NOT ONNX_MODELS_DIR)
  if (ROOT_ONNX_FOUND AND NOT broken_onnx)
    set(ONNX_MODELS_DIR ${CMAKE_CURRENT_BINARY_DIR}/input_models)
    set(generate_onnx_models TRUE)
    # Get the model names from the generator script (--list does not require
    # the onnx module, only a Python interpreter).
    execute_process(
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_input_models.py --list
      OUTPUT_VARIABLE _onnx_model_names
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _onnx_list_status)
    if (NOT _onnx_list_status EQUAL 0)
      message(FATAL_ERROR "Failed to list the SOFIE ONNX test models with generate_input_models.py")
    endif()
    string(REPLACE "\n" ";" ONNX_MODEL_NAMES "${_onnx_model_names}")
  else()
    message(WARNING "The onnx Python module is not usable: the ONNX input models for the TMVA SOFIE tests cannot be generated and the corresponding tests are disabled.")
  endif()
else()
  get_filename_component(ONNX_MODELS_DIR ${ONNX_MODELS_DIR} ABSOLUTE)
  file(GLOB ONNX_FILES "${ONNX_MODELS_DIR}/*.onnx")
  set(ONNX_MODEL_NAMES "")
  foreach(onnx_file ${ONNX_FILES})
    get_filename_component(fname ${onnx_file} NAME_WE)
    list(APPEND ONNX_MODEL_NAMES ${fname})
  endforeach()
endif()

if (ONNX_MODELS_DIR)
  # Creating the appropriate code to parse all ONNX model files. It is much
  # faster to combine all parsing in a single executable which will avoid
  # initialization time (especially when using ROOT)
  set(CAPTURE_STR "EmitModel( \"@1\", \"@2\");")
  set(ALL_CAPTURES "")
  foreach(fname ${ONNX_MODEL_NAMES})
    string(REPLACE "@1" ${ONNX_MODELS_DIR}/${fname}.onnx cap ${CAPTURE_STR})
    string(REPLACE "@2" ${fname} cap ${cap})
    list(APPEND ALL_CAPTURES ${cap})
  endforeach()
  string(REPLACE ";" ";\n" EMIT_CAPTURES "${ALL_CAPTURES}")
  configure_file(EmitFromONNX.cxx.in EmitFromONNX_all.cxx @ONLY)

  ROOTTEST_GENERATE_EXECUTABLE(emitFromONNX EmitFromONNX_all.cxx
                               LIBRARIES protobuf::libprotobuf ROOTTMVASofie ROOTTMVASofieParser
                               FIXTURES_SETUP sofie-compile-models-onnx-build)

  # silence protobuf warnings seen in version 3.0 and 3.6. Not needed from protobuf version 3.17
  target_compile_options(emitFromONNX PRIVATE -Wno-unused-parameter -Wno-array-bounds)

  set(compile_models_fixtures sofie-compile-models-onnx-build)
  if (generate_onnx_models)
    ROOTTEST_ADD_TEST(SofieGenerateModels_ONNX
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_input_models.py --outdir ${ONNX_MODELS_DIR}
      FIXTURES_SETUP sofie-generate-models-onnx
    )
    list(APPEND compile_models_fixtures sofie-generate-models-onnx)
  endif()

  ROOTTEST_ADD_TEST(SofieCompileModels_ONNX
    COMMAND ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1 ./emitFromONNX
    FIXTURES_REQUIRED ${compile_models_fixtures}
    FIXTURES_SETUP sofie-compile-models-onnx
  )

  # Creating a Google Test
  if (BLAS_FOUND)  # we need BLAS for compiling the models
    ROOT_EXECUTABLE(TestCustomModelsFromONNX TestCustomModelsFromONNX.cxx
      LIBRARIES Core GTest::gtest GTest::gtest_main
    )
    ROOTTEST_ADD_TEST(TestCustomModelsFromONNX
                      EXEC ./TestCustomModelsFromONNX
                      FIXTURES_REQUIRED sofie-compile-models-onnx)

    if (clad)
      ROOT_EXECUTABLE(TestCladAutodiff TestCladAutodiff.cxx
        LIBRARIES Core GTest::gtest GTest::gtest_main
      )
      ROOTTEST_ADD_TEST(TestCladAutodiff
                        EXEC ./TestCladAutodiff
                        FIXTURES_REQUIRED sofie-compile-models-onnx)
    endif()
  endif()
endif()

if (BLAS_FOUND)
  if (clad)
    # Creating a Google Test for the automatic differentiation of Gemm_Call
    ROOT_ADD_GTEST(TestGemmDerivative TestGemmDerivative.cxx LIBRARIES Core BLAS::BLAS)
  endif()
endif()

if (ROOT_TORCH_FOUND AND ROOT_ONNX_FOUND AND NOT broken_onnx)
  # ModelGeneratorUtils.py holds the code shared by the generators below, so it
  # has to be copied next to them.
  configure_file(ModelGeneratorUtils.py ModelGeneratorUtils.py COPYONLY)
  foreach(generator Conv1d Conv2d Conv3d ConvTrans2d Linear Recurrent)
    configure_file(${generator}ModelGenerator.py ${generator}ModelGenerator.py COPYONLY)
  endforeach()

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
