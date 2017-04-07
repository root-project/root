# - Locate vdt library
# Defines:
#
#  VDT_FOUND
#  VDT_INCLUDE_DIRS
#  VDT_LIBRARIES

find_path(VDT_INCLUDE_DIR NAMES vdt/vdtMath.h  HINTS ${VDT_DIR}/include /usr/include/vdt)
find_library(VDT_LIBRARY NAMES vdt HINTS ${VDT_DIR}/lib/ $ENV{VDT_DIR}/lib/ /usr/lib)

set(VDT_INCLUDE_DIRS ${VDT_INCLUDE_DIR})
set(VDT_LIBRARIES ${VDT_LIBRARY})


# handle the QUIETLY and REQUIRED arguments and set VDT_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(VDT DEFAULT_MSG VDT_INCLUDE_DIR VDT_LIBRARY)

mark_as_advanced(VDT_FOUND VDT_INCLUDE_DIRS VDT_LIBRARIES)
