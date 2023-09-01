# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#.rst:
# FindAfterImage (http://www.afterstep.org/afterimage)
# -------
#
# Find the AfterImage library header and define variables.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   AFTERIMAGE_FOUND          - True if AfterImage is found
#   AFTERIMAGE_VERSION        - The version of AfterImage found (x.y)
#   AFTERIMAGE_INCLUDE_DIR    - Where to find afterimage.h
#   AFTERIMAGE_LIBRARIES      - Libraries to link against to use libAfterImage

find_program(AFTERIMAGE_CONFIG_EXECUTABLE afterimage-config)
mark_as_advanced(AFTERIMAGE_CONFIG_EXECUTABLE)

if(AFTERIMAGE_CONFIG_EXECUTABLE)
  execute_process(COMMAND ${AFTERIMAGE_CONFIG_EXECUTABLE} --version
    OUTPUT_VARIABLE AFTERIMAGE_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${AFTERIMAGE_CONFIG_EXECUTABLE} --cflags
    OUTPUT_VARIABLE AFTERIMAGE_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${AFTERIMAGE_CONFIG_EXECUTABLE} --libs
    OUTPUT_VARIABLE AFTERIMAGE_LIBRARIES OUTPUT_STRIP_TRAILING_WHITESPACE)

  separate_arguments(AFTERIMAGE_CFLAGS)
  string(REGEX MATCH "-I[^;]+" AFTERIMAGE_INCLUDE_DIR "${AFTERIMAGE_CFLAGS}")
  string(REPLACE "-I" "" AFTERIMAGE_INCLUDE_DIR "${AFTERIMAGE_INCLUDE_DIR}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AfterImage
  REQUIRED_VARS AFTERIMAGE_INCLUDE_DIR AFTERIMAGE_LIBRARIES VERSION_VAR AFTERIMAGE_VERSION)
