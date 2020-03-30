# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

if (require-python-deps)
    set(ROOT_PY_FAIL_ON_MISSING_FLAG REQUIRED)
else ()
    set(ROOT_PY_FAIL_ON_MISSING_FLAG QUIET)
endif()

find_python_module("numba" ${ROOT_PY_FAIL_ON_MISSING_FLAG})
find_python_module("pandas" ${ROOT_PY_FAIL_ON_MISSING_FLAG})
