# - Try to find libcmaes, https://github.com/beniz/libcmaes
# Once done this will define
#
#  LIBCMAES_FOUND - system has libcmaes
#  LIBCMES_INCLUDE_DIR - the libcmaes include directory
#  LIBCMAES_LIBRARIES - Link these to use libcmaes
#  LIBCMAES_DEFINITIONS - Compiler switches required for using libcmaes
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#


# Copyright (c) 2014, Emmanuel Benazera, <emmanuel.benazera@lri.fr>
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.

if ( LIBCMAES_INCLUDE_DIR AND LIBCMAES_LIBRARIES )
   # in cache already
   SET(Libcmaes_FIND_QUIETLY TRUE)
endif ( LIBCMAES_INCLUDE_DIR AND LIBCMAES_LIBRARIES )

# use pkg-config to get the directories and then use these values
# in the FIND_PATH() and FIND_LIBRARY() calls
if( NOT WIN32 )
  find_package(PkgConfig)

  pkg_check_modules(PC_LIBCMAES REQUIRED libcmaes)

  set(LIBCMAES_INCLUDE_DIR ${PC_LIBCMAES_INCLUDE_DIRS})
  set(LIBCMAES_LIBRARIES ${PC_LIBCMAES_LIBRARY_DIRS})
  set(LIBCMAES_DEFINITIONS ${PC_LIBCMAES_CFLAGS_OTHER})
endif( NOT WIN32 )

find_path(LIBCMAES_INCLUDE_DIR NAMES cmaes.h
  PATHS
  ${PC_LIBCMAES_INCLUDEDIR}
  ${PC_LIBCMAES_INCLUDE_DIRS}
)

find_library(LIBCMAES_LIBRARIES NAMES libcmaes
  PATHS
  ${PC_LIBCMAES_LIBDIR}
  ${PC_LIBCMAES_LIBRARY_DIRS}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Libcmaes DEFAULT_MSG LIBCMAES_INCLUDE_DIR LIBCMAES_LIBRARIES )

# show the LIBCMAES_INCLUDE_DIR and LIBCMAES_LIBRARIES variables only in the advanced view
mark_as_advanced(LIBCMAES_INCLUDE_DIR LIBCMAES_LIBRARIES )
