# - Locate dCache library
# Defines:
#
#  DCAP_FOUND
#  DCAP_INCLUDE_DIR
#  DCAP_INCLUDE_DIRS (not cached)
#  DCAP_LIBRARIES

find_path(DCAP_INCLUDE_DIR NAMES dcap.h  HINTS ${DCAP_DIR}/include $ENV{DCAP_DIR}/include)
find_library(DCAP_LIBRARY NAMES dcap HINTS ${DCAP_DIR}/lib $ENV{DCAP_DIR}/lib)

set(DCAP_INCLUDE_DIRS ${DCAP_INCLUDE_DIR})
set(DCAP_LIBRARIES ${DCAP_LIBRARY})


# handle the QUIETLY and REQUIRED arguments and set DCAP_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(DCAP DEFAULT_MSG DCAP_INCLUDE_DIR DCAP_LIBRARY)

mark_as_advanced(DCAP_FOUND DCAP_INCLUDE_DIR DCAP_LIBRARY)
