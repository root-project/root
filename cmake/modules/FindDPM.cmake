# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# - Locate dpm library
# Defines:
#
#  DPM_FOUND
#  DPM_INCLUDE_DIR
#  DPM_INCLUDE_DIRS (not cached)
#  DPM_LIBRARIES

set(DPM_FOUND FALSE)

find_path(DPM_INCLUDE_DIR NAMES dpm_api.h HINTS ${DPM_DIR}/include $ENV{DPM_DIR}/include /usr/include PATH_SUFFIXES dpm)
find_library(DPM_dpm_LIBRARY NAMES dpm HINTS ${DPM_DIR}/lib $ENV{DPM_DIR}/lib)
find_library(DPM_lcgdm_LIBRARY NAMES lcgdm HINTS ${DPM_DIR}/lib $ENV{DPM_DIR}/lib)

set(DPM_INCLUDE_DIRS ${DPM_INCLUDE_DIR})
set(DPM_LIBRARIES ${DPM_dpm_LIBRARY} ${DPM_lcgdm_LIBRARY})

if (DPM_INCLUDE_DIR AND DPM_dpm_LIBRARY AND DPM_lcgdm_LIBRARY)
  set(DPM_FOUND TRUE)
  message(STATUS "Found DPM at: ${DPM_LIBRARIES}")
endif()

mark_as_advanced(DPM_FOUND DPM_INCLUDE_DIR DPM_dpm_LIBRARY DPM_lcgdm_LIBRARY)
