# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

if(pyroot)
  add_subdirectory(pyroot)
  add_subdirectory(jupyroot)

  if(dataframe)
    message(STATUS "Distributed RDataFrame enabled")
    add_subdirectory(experimental/distrdf)
  else()
    message(STATUS "Requirements to enable distributed RDataFrame:")
    message(STATUS "    dataframe: required:ON, actual:${dataframe}")
  endif()
endif()

if(tpython)
  add_subdirectory(tpython)
endif()

if(r)
  add_subdirectory(r)
endif()
