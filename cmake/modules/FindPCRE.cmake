# Find the PCRE includes and library.
#
# This module defines
# PCRE_INCLUDE_DIR, where to locate PCRE header files
# PCRE_LIBRARIES, the libraries to link against to use PCRE
# PCRE_FOUND.  If false, you cannot build anything that requires PCRE.

set(_PCRE_PATHS ${PCRE_DIR} $ENV{PCRE_DIR})

find_path(PCRE_INCLUDE_DIR pcre.h HINTS ${_PCRE_PATHS} PATH_SUFFIXES pcre)
find_library(PCRE_PCRE_LIBRARY NAMES pcre HINTS ${_PCRE_PATHS})
find_library(PCRE_PCREPOSIX_LIBRARY NAMES pcreposix HINTS ${_PCRE_PATHS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PCRE DEFAULT_MSG PCRE_INCLUDE_DIR PCRE_PCRE_LIBRARY)
mark_as_advanced(PCRE_INCLUDE_DIR PCRE_PCREPOSIX_LIBRARY PCRE_PCRE_LIBRARY)
set(PCRE_LIBRARIES ${PCRE_PCRE_LIBRARY})
if(PCRE_PCREPOSIX_LIBRARY)
  list(APPEND PCRE_LIBRARIES ${PCRE_PCREPOSIX_LIBRARY})
endif()

