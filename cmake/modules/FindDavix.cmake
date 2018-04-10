#.rst:
# FindDavix
# -------
#
# Find the Davix library header and define variables.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines :prop_tgt:`IMPORTED` target ``Davix::Davix``,
# if Davix has been found
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   Davix_FOUND             - True if Davix is found.
#   Davix_INCLUDE_DIRS      - Where to find davix.hpp
#   Davix_INCLUDE_LIBRARIES - Where to find libdavix
#
# ::
#
#   Davix_VERSION        - The version of Davix found (x.y.z)
#

find_package(PkgConfig REQUIRED)

if(${Davix_FIND_REQUIRED})
  set(Davix_REQUIRED REQUIRED)
endif()

pkg_check_modules(DAVIX ${Davix_REQUIRED} davix>=${Davix_FIND_VERSION})

if(Davix_FOUND AND NOT TARGET Davix::Davix)
  add_library(Davix::Davix UNKNOWN IMPORTED)
  set_target_properties(Davix::Davix PROPERTIES
    IMPORTED_LOCATION "${Davix_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${Davix_INCLUDE_DIRS}")
endif()
