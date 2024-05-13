# - Try to find DAOS
# Once done this will define
#  DAOS_FOUND - System has DAOS
#  DAOS_INCLUDE_DIRS - The DAOS include directories
#  DAOS_LIBRARIES - The libraries needed to use DAOS
#
# Path to a local installation can be specified through -DCMAKE_PREFIX_PATH=

# Uncomment when DAOS provides .pc files
#find_package(PkgConfig)
#pkg_check_modules(PC_DAOS daos)

find_path(DAOS_INCLUDE_DIR daos.h
  HINTS ${PC_DAOS_INCLUDEDIR} ${PC_DAOS_INCLUDE_DIRS}
  PATHS /usr/local/include /usr/include include)

find_library(DAOS_LIBRARY NAMES daos
  HINTS ${PC_DAOS_LIBDIR} ${PC_DAOS_LIBRARY_DIRS}
  PATHS /usr/local/lib64 /usr/local/lib /usr/lib64 /usr/lib lib64 lib)
find_library(DAOS_common_LIBRARY NAMES daos_common
  HINTS ${PC_DAOS_LIBDIR} ${PC_DAOS_LIBRARY_DIRS}
  PATHS /usr/local/lib64 /usr/local/lib /usr/lib64 /usr/lib lib64 lib)
find_library(DAOS_gurt_LIBRARY NAMES gurt
  HINTS ${PC_DAOS_LIBDIR} ${PC_DAOS_LIBRARY_DIRS}
  PATHS /usr/local/lib64 /usr/local/lib /usr/lib64 /usr/lib lib64 lib)

set(DAOS_INCLUDE_DIRS ${DAOS_INCLUDE_DIR})
set(DAOS_LIBRARIES ${DAOS_LIBRARY} ${DAOS_common_LIBRARY} ${DAOS_gurt_LIBRARY})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set DAOS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(DAOS DEFAULT_MSG
                                  DAOS_INCLUDE_DIR DAOS_LIBRARY DAOS_common_LIBRARY DAOS_gurt_LIBRARY)

mark_as_advanced(DAOS_INCLUDE_DIR DAOS_LIBRARY DAOS_common_LIBRARY DAOS_gurt_LIBRARY)

