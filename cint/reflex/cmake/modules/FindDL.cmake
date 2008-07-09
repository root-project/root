# - Find dl functions
# This module finds dl libraries.
#
# It sets the following variables:
#  DL_FOUND       - Set to false, or undefined, if dl libraries aren't found.
#  DL_INCLUDE_DIR - The dl include directory.
#  DL_LIBRARY     - The dl library to link against.

INCLUDE(CheckFunctionExists)

FIND_PATH(DL_INCLUDE_DIR NAMES dlfcn.h)
FIND_LIBRARY(DL_LIBRARY NAMES dl)

IF (DL_LIBRARY)
   SET(DL_FOUND TRUE)
ELSE (DL_LIBRARY)
   # if dlopen can be found without linking in dl then,
   # dlopen is part of libc, so don't need to link extra libs.
   CHECK_FUNCTION_EXISTS(dlopen DL_FOUND)
   SET(DL_LIBRARY "")
ENDIF (DL_LIBRARY)

IF (DL_FOUND)

   # show which dl was found only if not quiet
   IF (NOT DL_FIND_QUIETLY)
      MESSAGE(STATUS "Found dl: ${DL_LIBRARY}")
   ENDIF (NOT DL_FIND_QUIETLY)

ELSE (DL_FOUND)

   # fatal error if dl is required but not found
   IF (DL_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find dl")
   ENDIF (DL_FIND_REQUIRED)

ENDIF (DL_FOUND)
