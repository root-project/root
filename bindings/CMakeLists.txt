# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

if(python)
  if(NOT pyroot_experimental)
    add_subdirectory(pyroot)
  else()
    add_subdirectory(pyroot_experimental)
    add_subdirectory(tpython)
    add_subdirectory(jupyroot)
    add_subdirectory(jsmva)
  endif()
endif()

if(r)
  add_subdirectory(r)
endif()
