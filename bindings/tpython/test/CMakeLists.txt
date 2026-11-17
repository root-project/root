# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Jonas Rembser CERN, 2024

# Even for the C++ tests, we need to setup the right environment to use PyROOT,
# because we are using PyROOT from C++. This environment mimics the one used in
# ROOT_ADD_PYUNITTEST.
if(MSVC)
  set(tpython_gtest_env
      PYTHONPATH=${ROOTSYS}/bin;$ENV{PYTHONPATH})
else()
  set(tpython_gtest_env
      PYTHONPATH=${ROOTSYS}/lib:$ENV{PYTHONPATH})
endif()

if(NOT MSVC)
    # These tests fail on Windows because of a problem with std::any
  # input_line_33:7:52: error: address of overloaded function 'make_any' does not match required type 'std::any (int &&)'
  #     new (ret) (std::any) (((std::any (&)(int &&))std::make_any<int, int, 0>)((int&&)*(int*)args[0]));
  ROOT_ADD_GTEST(testTPython testTPython.cxx LIBRARIES ROOTTPython ENVIRONMENT ${tpython_gtest_env})
  # Use a separate process because the other TPython tests initialize Python successfully.
  ROOT_ADD_GTEST(testTPythonInitialization testTPythonInitialization.cxx
    LIBRARIES ROOTTPython
    ENVIRONMENT ${tpython_gtest_env} PYTHONHOME=${CMAKE_CURRENT_BINARY_DIR}/nonexistent-python-home)
  ROOT_ADD_PYUNITTEST(test_tpython test_tpython.py)
endif()

# Tests using TPython from ROOT macros and from the C++ Jupyter kernel
set(ROOT_root_CMD $<TARGET_FILE:root.exe>)

# A test for issue #9058
ROOTTEST_ADD_TEST(execscript
                  MACRO execscript.C
                  COPY_TO_BUILDDIR execscript.py
                  FAILREGEX "FAILURE")

# On Windows, code JIT-compiled in the interpreter cannot resolve the
# CppInternal::DispatchRaw::MakeFunctionCallable_interp dispatch slot: it
# is a data symbol (a function-pointer variable), which the bindexplib
# generated export tables do not cover. Exporting writable data across a
# DLL boundary needs a considered fix (or an upstream CppInterOp change
# to register the dispatch slots by value instead of by symbol lookup).
if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(api
                    MACRO runPyAPITestCppyy.C
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyAPITest.ref)
endif()

if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(class
                    MACRO runPyClassTest.C
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyClassTest.ref)
endif()

ROOTTEST_ADD_TEST(notebook
                  COPY_TO_BUILDDIR tpython.ipynb
                  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/roottest/python/JupyROOT/nbdiff.py tpython.ipynb
                  RUN_SERIAL
                  PYTHON_DEPS jupyter)
