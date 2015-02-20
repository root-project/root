# - Locate chirp (from cctools) library
# Defines:
#
#  CHIRP_FOUND
#  CHIRP_INCLUDE_DIR
#  CHIRP_INCLUDE_DIRS (not cached)
#  CHIRP_LIBRARIES

find_path(CHIRP_INCLUDE_DIR NAMES chirp.h  HINTS ${CHIRP_DIR}/include/cctools $ENV{CHIRP_DIR}/include/cctools)
find_library(CHIRP_LIBRARY NAMES chirp_client HINTS ${CHIRP_DIR}/lib $ENV{CHIRP_DIR}/lib)

set(CHIRP_INCLUDE_DIRS ${CHIRP_INCLUDE_DIR})
set(CHIRP_LIBRARIES ${CHIRP_LIBRARY})


# handle the QUIETLY and REQUIRED arguments and set CHIRP_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CHIRP DEFAULT_MSG CHIRP_INCLUDE_DIR CHIRP_LIBRARY)

mark_as_advanced(CHIRP_FOUND CHIRP_INCLUDE_DIR CHIRP_LIBRARY)
