#.rst:
# FindDavix
# -------
#
# Find Davix library for file management over HTTP-based protocols.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines :prop_tgt:`IMPORTED` target:
#
# ``Davix::Davix``
#   The libdavix library, if found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``DAVIX_FOUND``
#   True if Davix has been found.
# ``DAVIX_INCLUDE_DIRS``
#   Where to find davix.hpp, etc.
# ``DAVIX_LIBRARIES``
#   The libraries to link against to use Davix.
# ``DAVIX_VERSION``
#   The version of the Davix library found (e.g. 0.6.4)
#
# Obsolete variables
# ^^^^^^^^^^^^^^^^^^
#
# The following variables may also be set, for backwards compatibility:
#
# ``DAVIX_LIBRARY``
#   where to find the DAVIX library.
# ``DAVIX_INCLUDE_DIR``
#   where to find the DAVIX headers (same as DAVIX_INCLUDE_DIRS)
#

foreach(var FOUND INCLUDE_DIR INCLUDE_DIRS LIBRARY LIBRARIES)
  unset(DAVIX_${var} CACHE)
endforeach()

find_package(PkgConfig)

if(PKG_CONFIG_FOUND)
  if(${Davix_FIND_REQUIRED})
    set(Davix_REQUIRED REQUIRED)
  endif()

  pkg_check_modules(DAVIX ${Davix_REQUIRED} davix>=${Davix_FIND_VERSION})

  set(DAVIX_LIBRARIES ${DAVIX_LDFLAGS})
  set(DAVIX_LIBRARY ${DAVIX_LIBRARIES})
  set(DAVIX_INCLUDE_DIRS ${DAVIX_INCLUDE_DIRS})
  set(DAVIX_INCLUDE_DIR ${DAVIX_INCLUDE_DIRS})
endif()

if(DAVIX_FOUND AND NOT TARGET Davix::Davix)
  add_library(Davix::Davix INTERFACE IMPORTED)
  set_property(TARGET Davix::Davix PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${DAVIX_INCLUDE_DIRS}")
  set_property(TARGET Davix::Davix PROPERTY INTERFACE_LINK_LIBRARIES "${DAVIX_LIBRARIES}")
endif()

mark_as_advanced(DAVIX_INCLUDE_DIR DAVIX_LIBRARY)
