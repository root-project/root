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
  set(tpython_gtest_env ROOTSYS=${ROOTSYS}
      PYTHONPATH=${ROOTSYS}/bin;$ENV{PYTHONPATH})
else()
  set(tpython_gtest_env ROOTSYS=${ROOTSYS}
      PATH=${ROOTSYS}/bin:$ENV{PATH}
      LD_LIBRARY_PATH=${ROOTSYS}/lib:$ENV{LD_LIBRARY_PATH}
      PYTHONPATH=${ROOTSYS}/lib:$ENV{PYTHONPATH})
endif()

if(NOT MSVC)
    # These tests fail on Windows because of a problem with std::any
  # input_line_33:7:52: error: address of overloaded function 'make_any' does not match required type 'std::any (int &&)'
  #     new (ret) (std::any) (((std::any (&)(int &&))std::make_any<int, int, 0>)((int&&)*(int*)args[0]));
  ROOT_ADD_GTEST(testTPython testTPython.cxx LIBRARIES ROOTTPython ENVIRONMENT ${tpython_gtest_env})
  ROOT_ADD_PYUNITTEST(test_tpython test_tpython.py)
endif()
