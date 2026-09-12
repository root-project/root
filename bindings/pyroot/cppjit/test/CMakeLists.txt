# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# The suite is pytest-based with no unittest-compatible entry points, so the
# tests run through `python -m pytest` directly. Dictionaries are compiled by
# ACLiC into the test working directory (the suite's own Makefile is disabled
# via CPPJIT_TEST_SKIP_MAKE).

set(pythonpaths ${localruntimedir} $ENV{PYTHONPATH})
cmake_path(CONVERT "${pythonpaths}" TO_NATIVE_PATH_LIST pythonpaths_native)

set(common_env PYTHONPATH=${pythonpaths_native} CPPJIT_TEST_SKIP_MAKE=1)
if(NOT MSVC)
  list(APPEND common_env ${ld_library_path}=${CMAKE_CURRENT_BINARY_DIR})
endif()

# Some dictionary sources include Python.h (e.g. pythonizables.cxx), which
# ACLiC does not find on its own.
set(root_cmd $<TARGET_FILE:root.exe> -b -q
    -e "gSystem->AddIncludePath(\"-I${Python3_INCLUDE_DIRS}\")")
if(APPLE)
  # ACLiC links dictionaries as -dynamiclib; CPython symbols in them resolve
  # from the hosting interpreter process, as for any extension module.
  list(APPEND root_cmd
    -e "gSystem->AddLinkedLibs(\"-Wl,-undefined,dynamic_lookup\")")
elseif(MSVC)
  # Python.h auto-links pythonNN.lib via #pragma comment(lib, ...); ACLiC's
  # def/import-lib step honors the directive but has no library search path,
  # so suppress it and hand the link step the library explicitly.
  list(APPEND root_cmd
    -e "gSystem->AddIncludePath(\"-DPy_NO_LINK_LIB\")"
    -e "gSystem->AddLinkedLibs(\"${Python3_LIBRARIES}\")")
endif()

function(CPPJIT_ADD_PYTEST name)
  cmake_parse_arguments(ARG
    "" "" "DICTS;FIXTURES_SETUP;FIXTURES_REQUIRED" ${ARGN})

  set(extra)
  if(ARG_DICTS)
    set(copy_files)
    set(precmd_args)
    foreach(dict ${ARG_DICTS})
      list(APPEND copy_files
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/${dict}.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/${dict}.h)
      list(APPEND precmd_args -e ".L ${dict}.cxx++")
    endforeach()
    list(APPEND extra COPY_TO_BUILDDIR ${copy_files}
                      PRECMD ${root_cmd} ${precmd_args})
  endif()

  string(REGEX REPLACE "[_]" "-" good_name "${name}")
  set(test_name pyunittests-bindings-cppjit-${good_name})

  ROOT_ADD_TEST(${test_name}
    # Upstream marks platform xfails without explicit strict=; the in-tree
    # patches fix many of them on the ROOT stack, so an xpass must not fail
    # the suite (pyproject.toml sets xfail_strict=true for upstream CI).
    COMMAND ${Python3_EXECUTABLE} -B -m pytest -v -p no:cacheprovider
            -o xfail_strict=false
            ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py
    WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}
    ENVIRONMENT ${common_env}
    LABELS python
    PYTHON_DEPS pytest
    ${extra})

  if(ARG_FIXTURES_SETUP)
    set_property(TEST ${test_name} PROPERTY FIXTURES_SETUP ${ARG_FIXTURES_SETUP})
  endif()
  if(ARG_FIXTURES_REQUIRED)
    set_property(TEST ${test_name} PROPERTY FIXTURES_REQUIRED ${ARG_FIXTURES_REQUIRED})
  endif()
endfunction()

CPPJIT_ADD_PYTEST(test_conversions DICTS conversions)
if(NOT MSVC)
  # Windows ACLiC emits a .rootmap whose forward declaration of the
  # std::hash<StructWithHash> specialization conflicts with the just-parsed
  # header; the failed load makes ACLiC relink over its own locked DLL.
  CPPJIT_ADD_PYTEST(test_cpp11features DICTS cpp11features)
endif()
CPPJIT_ADD_PYTEST(test_crossinheritance DICTS crossinheritance)
CPPJIT_ADD_PYTEST(test_doc_features DICTS doc_helper)
CPPJIT_ADD_PYTEST(test_fragile DICTS fragile)
CPPJIT_ADD_PYTEST(test_operators DICTS operators)
CPPJIT_ADD_PYTEST(test_overloads DICTS overloads)
CPPJIT_ADD_PYTEST(test_pythonization DICTS pythonizables)
CPPJIT_ADD_PYTEST(test_stltypes DICTS stltypes)
CPPJIT_ADD_PYTEST(test_streams DICTS std_streams)
CPPJIT_ADD_PYTEST(test_templates DICTS templates)

CPPJIT_ADD_PYTEST(test_pythonify DICTS example01
  FIXTURES_SETUP cppjit_example01_dict_fixture)
CPPJIT_ADD_PYTEST(test_advancedcpp DICTS advancedcpp advancedcpp2)
CPPJIT_ADD_PYTEST(test_lowlevel DICTS datatypes
  FIXTURES_SETUP cppjit_datatypes_dict_fixture)

CPPJIT_ADD_PYTEST(test_aclassloader
  FIXTURES_REQUIRED cppjit_example01_dict_fixture)
CPPJIT_ADD_PYTEST(test_basic_api
  FIXTURES_REQUIRED cppjit_example01_dict_fixture)
CPPJIT_ADD_PYTEST(test_datatypes
  FIXTURES_REQUIRED cppjit_datatypes_dict_fixture)

CPPJIT_ADD_PYTEST(test_api)
CPPJIT_ADD_PYTEST(test_boost)
CPPJIT_ADD_PYTEST(test_concurrent)
CPPJIT_ADD_PYTEST(test_cpp23features)
CPPJIT_ADD_PYTEST(test_eigen)
CPPJIT_ADD_PYTEST(test_leakcheck)
CPPJIT_ADD_PYTEST(test_numba)
CPPJIT_ADD_PYTEST(test_regression)
