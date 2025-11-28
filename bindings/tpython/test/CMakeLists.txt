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

ROOT_ADD_GTEST(testTPython testTPython.cxx LIBRARIES ROOTTPython ENVIRONMENT ${tpython_gtest_env})

if(NOT MSVC)
  # This test fails on Windows because MSVC’s standard library does not export
  # the symbols needed for cling to JIT-compile code involving std::any,
  # causing symbol resolution to fail.
  ROOT_ADD_PYUNITTEST(test_tpython test_tpython.py)
endif()
