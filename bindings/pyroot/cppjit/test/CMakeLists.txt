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
    "DROP_ROOTMAP" "" "DICTS;FIXTURES_SETUP;FIXTURES_REQUIRED" ${ARGN})

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
    if(ARG_DROP_ROOTMAP AND MSVC)
      foreach(dict ${ARG_DICTS})
        list(APPEND precmd_args
          -e "(void)gSystem->Unlink(\"${dict}_cxx.rootmap\")")
      endforeach()
    endif()
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
# The rootmap Windows ACLiC emits forward-declares the variadic matryoshka
# template without its parameter pack; parsing it fails the whole dictionary
# payload. The tests load the library by name and never need the rootmap.
CPPJIT_ADD_PYTEST(test_templates DICTS templates DROP_ROOTMAP)

CPPJIT_ADD_PYTEST(test_pythonify DICTS example01
  FIXTURES_SETUP cppjit_example01_dict_fixture)
CPPJIT_ADD_PYTEST(test_advancedcpp DICTS advancedcpp advancedcpp2)
CPPJIT_ADD_PYTEST(test_lowlevel DICTS datatypes
  FIXTURES_SETUP cppjit_datatypes_dict_fixture)

# [TEMP-CI-PROBE] test_lowlevel passes on Windows x64, then the process
# fail-fasts at exit; the round-8 matrix pinned test07_3D_custom_struct as
# the sole poison. Phase-replica probes (TestPROBE07 in test_lowlevel.py)
# bisect within it: cppdef of the JIT global array, the constexpr global
# read, the eager proxy-tuple bind, the call loop. no:faulthandler keeps a
# crash from looping the fault handler (30GB log in round 8). The probes
# reuse the dictionary ACLiC-built by the main test (DEPENDS still runs
# after a failed dependency, unlike a fixture). Dropped with the fix.
if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8)
  # The crash under bisection can loop python's fault handler instead of
  # fail-fasting (rounds 8/12 produced multi-GB logs); disable it for the
  # main test as the probes already do.
  set_property(TEST pyunittests-bindings-cppjit-test-lowlevel APPEND
    PROPERTY ENVIRONMENT "PYTEST_ADDOPTS=-p no:faulthandler")
  set(lowlevel_probes
      "name-s|only:test_probe07e_name_s"
      "min-read|only:test_probe07i_min_read"
      "min-defonly|only:test_probe07j_min_defonly"
      "min-nonconst|only:test_probe07k_min_nonconst"
      "min-read-nodict|only:test_probe07l_min_read_nodict")
  foreach(probe ${lowlevel_probes})
    string(REPLACE "|" ";" probe "${probe}")
    list(GET probe 0 probe_id)
    list(GET probe 1 probe_spec)
    ROOT_ADD_TEST(pyunittests-bindings-cppjit-lowlevel-probe-${probe_id}
      COMMAND ${Python3_EXECUTABLE} -B -m pytest -v -p no:cacheprovider
              -p no:faulthandler -o xfail_strict=false
              ${CMAKE_CURRENT_SOURCE_DIR}/test_lowlevel.py
      WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}
      ENVIRONMENT ${common_env} CPPJIT_LOWLEVEL_PROBE=${probe_spec}
      LABELS python
      PYTHON_DEPS pytest
      DEPENDS pyunittests-bindings-cppjit-test-lowlevel)
  endforeach()
endif()

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
