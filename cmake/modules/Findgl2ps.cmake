# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# - Locate gl2ps library
# Defines:
#
#  gl2ps_FOUND
#  gl2ps_INCLUDE_DIR
#  gl2ps_INCLUDE_DIRS (not cached)
#  gl2ps_LIBRARIES

find_path(gl2ps_INCLUDE_DIR NAMES gl2ps.h HINTS ${gl2ps_DIR}/include $ENV{gl2ps_DIR}/include /usr/include)
find_library(gl2ps_LIBRARY NAMES gl2ps HINTS ${gl2ps_DIR}/lib $ENV{gl2ps_DIR}/lib)

set(gl2ps_INCLUDE_DIRS ${gl2ps_INCLUDE_DIR})
if(gl2ps_LIBRARY)
  set(gl2ps_LIBRARIES ${gl2ps_LIBRARY})
endif()


# handle the QUIETLY and REQUIRED arguments and set gl2ps_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(gl2ps DEFAULT_MSG gl2ps_LIBRARY gl2ps_INCLUDE_DIR)

mark_as_advanced(gl2ps_FOUND gl2ps_INCLUDE_DIR gl2ps_LIBRARY)

if(gl2ps_FOUND)
  if(NOT TARGET gl2ps::g2ps)
    add_library(gl2ps::gl2ps UNKNOWN IMPORTED)
    set_target_properties(gl2ps::gl2ps PROPERTIES
      IMPORTED_LOCATION "${gl2ps_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${gl2ps_INCLUDE_DIR}")
  endif()
endif()
