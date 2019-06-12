#.rst:
# FindCFITSIO
# -------
#
# Find the CFITSIO library header and define variables.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines :prop_tgt:`IMPORTED` target ``CFITSIO::CFITSIO``,
# if CFITSIO has been found
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   CFITSIO_FOUND          - True if CFITSIO is found.
#   CFITSIO_INCLUDE_DIRS   - Where to find fitsio.h
#
# ::
#
#   CFITSIO_VERSION        - The version of CFITSIO found (x.y)
#   CFITSIO_VERSION_MAJOR  - The major version of CFITSIO
#   CFITSIO_VERSION_MINOR  - The minor version of CFITSIO

find_path(CFITSIO_INCLUDE_DIR NAME fitsio.h
  PATH_SUFFIXES libcfitsio3 libcfitsio0 cfitsio
  PATHS $ENV{CFITSIO} ${CFITSIO_DIR}/include ${GNUWIN32_DIR}/include)

if(NOT CFITSIO_LIBRARY)
  find_library(CFITSIO_LIBRARY NAMES cfitsio
    PATHS $ENV{CFITSIO} ${CFITSIO_DIR}/lib ${GNUWIN32_DIR}/lib)
endif()

mark_as_advanced(CFITSIO_INCLUDE_DIR CFITSIO_LIBRARY)

if(CFITSIO_INCLUDE_DIR AND EXISTS "${CFITSIO_INCLUDE_DIR}/fitsio.h")
  file(STRINGS "${CFITSIO_INCLUDE_DIR}/fitsio.h" CFITSIO_H REGEX "^#define CFITSIO_[A-Z]+[ ]+[0-9]+.*$")
  string(REGEX REPLACE ".+CFITSIO_MAJOR[ ]+([0-9]+).*$"   "\\1" CFITSIO_VERSION_MAJOR "${CFITSIO_H}")
  string(REGEX REPLACE ".+CFITSIO_MINOR[ ]+([0-9]+).*$"   "\\1" CFITSIO_VERSION_MINOR "${CFITSIO_H}")
  set(CFITSIO_VERSION "${CFITSIO_VERSION_MAJOR}.${CFITSIO_VERSION_MINOR}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CFITSIO
  REQUIRED_VARS CFITSIO_LIBRARY CFITSIO_INCLUDE_DIR VERSION_VAR CFITSIO_VERSION)

if(CFITSIO_FOUND)
  set(CFITSIO_INCLUDE_DIRS "${CFITSIO_INCLUDE_DIR}")

  if(NOT CFITSIO_LIBRARIES)
    set(CFITSIO_LIBRARIES ${CFITSIO_LIBRARY})
    if(${CFITSIO_VERSION} VERSION_GREATER_EQUAL 3.42)
      find_package(CURL QUIET)
      if(CURL_FOUND)
        set(CFITSIO_LIBRARIES ${CFITSIO_LIBRARIES} ${CURL_LIBRARIES})
      endif()
    endif()
  endif()

  if(NOT TARGET CFITSIO::CFITSIO)
    add_library(CFITSIO::CFITSIO UNKNOWN IMPORTED)
    set_target_properties(CFITSIO::CFITSIO PROPERTIES
      IMPORTED_LOCATION "${CFITSIO_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CFITSIO_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${CURL_LIBRARIES}"
    )
  endif()
endif()
