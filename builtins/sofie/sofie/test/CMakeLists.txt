############################################################################
# CMakeLists.txt file for building the SOFIE tests, which run with ctest.
# @author Federico Sossai, Sanjiban Sengupta
#
# The generated models are compiled by the tests with the C++ compiler used to
# build them, see driver_runner.h.
############################################################################

if(NOT TARGET GTest::gtest OR NOT TARGET GTest::gtest_main)
  find_package(GTest REQUIRED)
endif()
include(GoogleTest)

find_package(Python3 COMPONENTS Interpreter)

# nlohmann_json is used by the safetensors test to validate the structure of the
# generated weight file.
if(NOT TARGET nlohmann_json::nlohmann_json)
  find_package(nlohmann_json QUIET)
endif()

# Look for the needed Python modules
function(sofie_find_python_module module)
  string(TOUPPER ${module} MODULE)
  set(SOFIE_${MODULE}_FOUND FALSE PARENT_SCOPE)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import ${module}; print(${module}.__version__)"
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _version
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(_status EQUAL 0)
      set(SOFIE_${MODULE}_FOUND TRUE PARENT_SCOPE)
      set(SOFIE_${MODULE}_VERSION ${_version} PARENT_SCOPE)
    endif()
  endif()
endfunction()

sofie_find_python_module(torch)

# onnx 1.19.0 has a bug that makes this version unusable:
# https://github.com/onnx/onnx/issues/7249
# 1.19.1 instead could be used
# In that case, we have to disable the "TestSofieModels" test,
# which imports onnx indirectly via torch.onnx
sofie_find_python_module(onnx)
if(SOFIE_ONNX_FOUND AND DEFINED SOFIE_ONNX_VERSION)
  if(SOFIE_ONNX_VERSION VERSION_EQUAL "1.19.0")
    message(WARNING "Found broken onnx version ${SOFIE_ONNX_VERSION} (see https://github.com/onnx/onnx/issues/7249). Some SOFIE tests will be disabled.")
    set(broken_onnx TRUE)
  endif()
endif()

# Configuration of the drivers that run the generated models
set(SOFIE_TEST_CXX_FLAGS "-std=c++${CMAKE_CXX_STANDARD}")
configure_file(test_config.h.in test_config.h @ONLY)

# Adds a test executable with Google Test. Each test case is registered as a test
# for ctest, so that they can run in parallel.
function(sofie_add_gtest name source)
  cmake_parse_arguments(ARG "" "" "LIBRARIES;FIXTURES_REQUIRED" ${ARGN})
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE SOFIE GTest::gtest GTest::gtest_main ${ARG_LIBRARIES})
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
  if(ARG_FIXTURES_REQUIRED)
    gtest_discover_tests(${name} TEST_PREFIX "${name}." PROPERTIES FIXTURES_REQUIRED ${ARG_FIXTURES_REQUIRED})
  else()
    gtest_discover_tests(${name} TEST_PREFIX "${name}.")
  endif()
endfunction()

# The ONNX input models are not checked into the repository: they are created
# by generate_input_models.py, which runs as the SofieGenerateModels_ONNX
# test that the other ONNX tests depend on. Set ONNX_MODELS_DIR to use
# pre-existing models from a custom directory instead.
if(NOT ONNX_MODELS_DIR)
  if(SOFIE_ONNX_FOUND AND NOT broken_onnx)
    set(ONNX_MODELS_DIR ${CMAKE_CURRENT_BINARY_DIR}/input_models)
    set(generate_onnx_models TRUE)
  else()
    message(WARNING "The onnx Python module is not usable: the ONNX input models for the SOFIE tests cannot be generated and the corresponding tests are disabled.")
  endif()
else()
  get_filename_component(ONNX_MODELS_DIR ${ONNX_MODELS_DIR} ABSOLUTE)
endif()

if(ONNX_MODELS_DIR)
  # Emits the code for all models in ONNX_MODELS_DIR, which it lists at runtime.
  # One executable for all models avoids paying the initialization time per model.
  add_executable(EmitFromONNX EmitFromONNX.cxx)
  target_link_libraries(EmitFromONNX PRIVATE SOFIE)

  set(compile_models_fixtures "")
  if(generate_onnx_models)
    add_test(NAME SofieGenerateModels_ONNX
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_input_models.py --outdir ${ONNX_MODELS_DIR})
    set_tests_properties(SofieGenerateModels_ONNX PROPERTIES FIXTURES_SETUP sofie-generate-models-onnx)
    list(APPEND compile_models_fixtures sofie-generate-models-onnx)
  endif()

  add_test(NAME SofieCompileModels_ONNX COMMAND EmitFromONNX ${ONNX_MODELS_DIR})
  set_tests_properties(SofieCompileModels_ONNX PROPERTIES
                       FIXTURES_SETUP sofie-compile-models-onnx)
  if(compile_models_fixtures)
    set_tests_properties(SofieCompileModels_ONNX PROPERTIES FIXTURES_REQUIRED "${compile_models_fixtures}")
  endif()

  # Tests of the alias tensors. They assert on the RModel itself rather than on
  # the code it generates, so they need the ONNX models but none of the headers
  # emitted from them.
  set(alias_test_fixtures "")
  if(generate_onnx_models)
    set(alias_test_fixtures sofie-generate-models-onnx)
  endif()
  sofie_add_gtest(TestSofieAlias TestSofieAlias.cxx FIXTURES_REQUIRED ${alias_test_fixtures})
  target_compile_definitions(TestSofieAlias PRIVATE SOFIE_ONNX_MODELS_DIR="${ONNX_MODELS_DIR}")

  sofie_add_gtest(TestCustomModelsFromONNX TestCustomModelsFromONNX.cxx FIXTURES_REQUIRED sofie-compile-models-onnx)
endif()

# Unit tests for the ONNX parser layer: the tests write the model files
# themselves as hand-crafted protobuf wire format, so they need no Python
sofie_add_gtest(TestSofieParser TestSofieParser.cxx)

# Unit tests for the safetensors weight file format. nlohmann_json is used by
# the test itself to validate the structure of the generated file.
if(TARGET nlohmann_json::nlohmann_json)
  sofie_add_gtest(TestSafetensorsWeights TestSafetensorsWeights.cxx
                      LIBRARIES nlohmann_json::nlohmann_json)
else()
  message(STATUS "nlohmann_json not found: the SOFIE safetensors test is disabled")
endif()

if(SOFIE_TORCH_FOUND AND SOFIE_ONNX_FOUND AND NOT broken_onnx)
  # ModelGeneratorUtils.py holds the code shared by the generators below, so it
  # has to be copied next to them.
  configure_file(ModelGeneratorUtils.py ModelGeneratorUtils.py COPYONLY)
  foreach(generator Conv1d Conv2d Conv3d ConvTrans2d Linear Recurrent)
    configure_file(${generator}ModelGenerator.py ${generator}ModelGenerator.py COPYONLY)
  endforeach()

  sofie_add_gtest(TestSofieModels TestSofieModels.cxx)
endif()
