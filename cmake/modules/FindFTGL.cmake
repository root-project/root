# - Locate FTGL library
# Defines:
#
#  FTGL_FOUND
#  FTGL_INCLUDE_DIR
#  FTGL_LIBRARY
#  FTGL_INCLUDE_DIRS (not cached)
#  FTGL_LIBRARIES (not cached)

find_path(FTGL_INCLUDE_DIR FTGL/ftgl.h
          HINTS $ENV{FTGL_ROOT_DIR}/include ${FTGL_ROOT_DIR}/include)

find_library(FTGL_LIBRARY NAMES ftgl
             HINTS $ENV{FTGL_ROOT_DIR}/lib ${FTGL_ROOT_DIR}/lib)

set(FTGL_INCLUDE_DIRS ${FTGL_INCLUDE_DIR})
set(FTGL_LIBRARIES ${FTGL_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FTGL DEFAULT_MSG FTGL_INCLUDE_DIR FTGL_LIBRARY)
mark_as_advanced(FTGL_FOUND FTGL_INCLUDE_DIR FTGL_LIBRARY)

