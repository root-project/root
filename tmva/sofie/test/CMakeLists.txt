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
  else()
    message(WARNING "The onnx Python module is not usable: the ONNX input models for the TMVA SOFIE tests cannot be generated and the corresponding tests are disabled.")
  endif()
else()
  get_filename_component(ONNX_MODELS_DIR ${ONNX_MODELS_DIR} ABSOLUTE)
endif()

if (ONNX_MODELS_DIR)
  # Emits the code for all models in ONNX_MODELS_DIR, which it lists at runtime.
  # One executable for all models avoids paying the initialization time per model.
  ROOT_EXECUTABLE(emitFromONNX EmitFromONNX.cxx
                  LIBRARIES ROOTTMVASofie ROOTTMVASofieParser)
  if(ROOT_NEED_STDCXXFS)
    target_link_libraries(emitFromONNX PRIVATE stdc++fs)
  endif()

  set(compile_models_fixtures "")
  if (generate_onnx_models)
    ROOTTEST_ADD_TEST(SofieGenerateModels_ONNX
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_input_models.py --outdir ${ONNX_MODELS_DIR}
      FIXTURES_SETUP sofie-generate-models-onnx
    )
    list(APPEND compile_models_fixtures sofie-generate-models-onnx)
  endif()

  ROOTTEST_ADD_TEST(SofieCompileModels_ONNX
    COMMAND ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1 ./emitFromONNX ${ONNX_MODELS_DIR}
    FIXTURES_REQUIRED ${compile_models_fixtures}
    FIXTURES_SETUP sofie-compile-models-onnx
  )

  # Tests of the alias tensors. They assert on the RModel itself rather than on
  # the code it generates, so they need the ONNX models but none of the headers
  # emitted from them.
  set(alias_test_fixtures "")
  if (generate_onnx_models)
    set(alias_test_fixtures sofie-generate-models-onnx)
  endif()
  ROOT_EXECUTABLE(TestSofieAlias TestSofieAlias.cxx
    LIBRARIES ROOTTMVASofie ROOTTMVASofieParser GTest::gtest GTest::gtest_main
  )
  target_compile_definitions(TestSofieAlias PRIVATE SOFIE_ONNX_MODELS_DIR="${ONNX_MODELS_DIR}")
  ROOTTEST_ADD_TEST(TestSofieAlias
                    EXEC ./TestSofieAlias
                    FIXTURES_REQUIRED ${alias_test_fixtures})

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
    ROOT_ADD_GTEST(TestGemmDerivative TestGemmDerivative.cxx LIBRARIES Core BLAS::BLAS ROOTTMVASofie)
  endif()
endif()

# Unit tests for the ONNX parser layer: the tests write the model files
# themselves as hand-crafted protobuf wire format, so they need no Python
ROOT_ADD_GTEST(TestSofieParser TestSofieParser.cxx
  LIBRARIES Core ROOTTMVASofie ROOTTMVASofieParser)

# Unit tests for the safetensors weight file format. nlohmann_json is used by
# the test itself to validate the structure of the generated file.
ROOT_ADD_GTEST(TestSafetensorsWeights TestSafetensorsWeights.cxx
  LIBRARIES Core ROOTTMVASofie ROOTTMVASofieParser nlohmann_json::nlohmann_json)

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

