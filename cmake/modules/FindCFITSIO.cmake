# - Try to find CFITSIO
# Once done this will define
#
#  CFITSIO_FOUND - system has CFITSIO
#  CFITSIO_INCLUDE_DIR - the CFITSIO include directory
#  CFITSIO_LIBRARY - the CFITSIO library
#  CFITSIO_LIBRARIES (not cached) - Link these to use CFITSIO
#  CFITSIO_VERSION_STRING - Human readable version number of cfitsio
#  CFITSIO_VERSION_MAJOR  - Major version number of cfitsio
#  CFITSIO_VERSION_MINOR  - Minor version number of cfitsio

# Copyright (c) 2006, Jasem Mutlaq <mutlaqja@ikarustech.com>
# Based on FindLibfacile by Carsten Niehaus, <cniehaus@gmx.de>
# Chnaged by Pere Mato
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.


find_path(CFITSIO_INCLUDE_DIR fitsio.h
  PATH_SUFFIXES libcfitsio3 libcfitsio0 cfitsio
  PATHS
  $ENV{CFITSIO}
  ${CFITSIO_DIR}/include
  ${GNUWIN32_DIR}/include
)

find_library(CFITSIO_LIBRARY NAMES cfitsio
  PATHS
  $ENV{CFITSIO}
  ${CFITSIO_DIR}/lib
  ${GNUWIN32_DIR}/lib
)
if(CFITSIO_LIBRARY)
  set(CFITSIO_LIBRARIES ${CFITSIO_LIBRARY})
endif()

# handle the QUIETLY and REQUIRED arguments and set CFITSIO_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CFITSIO DEFAULT_MSG CFITSIO_INCLUDE_DIR CFITSIO_LIBRARY)

mark_as_advanced(
  CFITSIO_LIBRARY
  CFITSIO_INCLUDE_DIR
)

if (CFITSIO_FOUND)
  # Find the version of the cfitsio header
  FILE(READ "${CFITSIO_INCLUDE_DIR}/fitsio.h" FITSIO_H)
  STRING(REGEX REPLACE ".*#define CFITSIO_VERSION[^0-9]*([0-9]+)\\.([0-9]+).*" "\\1.\\2" CFITSIO_VERSION_STRING "${FITSIO_H}")
  STRING(REGEX REPLACE "^([0-9]+)[.]([0-9]+)" "\\1" CFITSIO_VERSION_MAJOR ${CFITSIO_VERSION_STRING})
  STRING(REGEX REPLACE "^([0-9]+)[.]([0-9]+)" "\\2" CFITSIO_VERSION_MINOR ${CFITSIO_VERSION_STRING})
  message(STATUS "Found CFITSIO version: ${CFITSIO_VERSION_STRING}")
endif()
