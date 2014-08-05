# - Locate DAVIX library
# Defines:
#
#  DAVIX_FOUND
#  DAVIX_INCLUDE_DIR
#  DAVIX_INCLUDE_DIRS (not cached)
#  DAVIX_LIBRARIES

find_path(DAVIX_INCLUDE_DIR NAMES davix.hpp PATH_SUFFIXES davix HINTS ${DAVIX_DIR}/include $ENV{DAVIX_DIR}/include)
find_library(DAVIX_LIBRARY NAMES davix HINTS ${DAVIX_DIR}/lib $ENV{DAVIX_DIR}/lib64 ${DAVIX_DIR}/lib $ENV{DAVIX_DIR}/lib64)

set(DAVIX_INCLUDE_DIRS ${DAVIX_INCLUDE_DIR})
set(DAVIX_LIBRARIES ${DAVIX_LIBRARY})

# Handle the QUIETLY and REQUIRED arguments and set DAVIX_FOUND to TRUE if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DAVIX DEFAULT_MSG DAVIX_INCLUDE_DIRS DAVIX_LIBRARIES)

mark_as_advanced(DAVIX_FOUND DAVIX_INCLUDE_DIRS DAVIX_LIBRARIES)
