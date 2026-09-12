# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

ROOT_PYTHON_PACKAGE(cppjit
  SOURCES
  cppjit/_stdcpp_fix.py
  cppjit/__init__.py
  cppjit/_cpython_cppjit.py
  cppjit/_pythonization.py
  cppjit/_typemap.py
  cppjit/_version.py
  cppjit/interactive.py
  cppjit/ll.py
  cppjit/numba_ext.py
  cppjit/reflex.py
  cppjit/types.py
)
