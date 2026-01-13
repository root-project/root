# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set(common_args GENERIC PYTHON_DEPS pytest)

if(NOT MSVC)
  list(APPEND common_args ENVIRONMENT ${ld_library_path}=${CMAKE_CURRENT_BINARY_DIR})
endif()

set(root_cmd $<TARGET_FILE:root.exe> -b -q -l)

function(ADD_PYUNITTEST_ACLIC TESTNAME)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs DICTNAME)
  cmake_parse_arguments(ARG
    "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
  )

  ROOT_ADD_PYUNITTEST(${TESTNAME} ${TESTNAME}.py ${common_args}
    COPY_TO_BUILDDIR ${ARG_DICTNAME}.cxx ${ARG_DICTNAME}.h
    PRECMD ${root_cmd} -e ".L ${ARG_DICTNAME}.cxx++"
  )
endfunction()

ADD_PYUNITTEST_ACLIC(test_conversions DICTNAME conversions)
ADD_PYUNITTEST_ACLIC(test_cpp11features DICTNAME cpp11features)
ADD_PYUNITTEST_ACLIC(test_crossinheritance DICTNAME crossinheritance)
ADD_PYUNITTEST_ACLIC(test_doc_features DICTNAME doc_helper)
ADD_PYUNITTEST_ACLIC(test_fragile DICTNAME fragile)
ADD_PYUNITTEST_ACLIC(test_operators DICTNAME operators)
ADD_PYUNITTEST_ACLIC(test_overloads DICTNAME overloads)
ADD_PYUNITTEST_ACLIC(test_pythonization DICTNAME pythonizables)
ADD_PYUNITTEST_ACLIC(test_stltypes DICTNAME stltypes)
ADD_PYUNITTEST_ACLIC(test_streams DICTNAME std_streams)
ADD_PYUNITTEST_ACLIC(test_templates DICTNAME templates)

ROOT_ADD_PYUNITTEST(test_pythonify test_pythonify.py ${common_args}
  COPY_TO_BUILDDIR example01.cxx example01.h
  PRECMD ${root_cmd} -e ".L example01.cxx++"
  FIXTURES_SETUP example01_dict_fixture
)

ROOT_ADD_PYUNITTEST(test_advancedcpp test_advancedcpp.py ${common_args}
  COPY_TO_BUILDDIR advancedcpp.cxx advancedcpp.h advancedcpp2.cxx advancedcpp2.h
  PRECMD ${root_cmd} -e ".L advancedcpp.cxx++" -e ".L advancedcpp2.cxx++"
)

ROOT_ADD_PYUNITTEST(test_lowlevel test_lowlevel.py ${common_args}
  COPY_TO_BUILDDIR datatypes.cxx datatypes.h
  PRECMD ${root_cmd} -e ".L datatypes.cxx++"
  FIXTURES_SETUP datatypes_dict_fixture
)

ROOT_ADD_PYUNITTEST(test_aclassloader test_aclassloader.py ${common_args} FIXTURES_REQUIRED example01_dict_fixture)
ROOT_ADD_PYUNITTEST(test_api test_api.py ${common_args})
ROOT_ADD_PYUNITTEST(test_boost test_boost.py ${common_args})
ROOT_ADD_PYUNITTEST(test_concurrent test_concurrent.py ${common_args})
ROOT_ADD_PYUNITTEST(test_datatypes test_datatypes.py ${common_args} FIXTURES_REQUIRED datatypes_dict_fixture)
ROOT_ADD_PYUNITTEST(test_eigen test_eigen.py ${common_args})
ROOT_ADD_PYUNITTEST(test_leakcheck test_leakcheck.py ${common_args})
ROOT_ADD_PYUNITTEST(test_numba test_numba.py ${common_args})
ROOT_ADD_PYUNITTEST(test_regression test_regression.py ${common_args})
