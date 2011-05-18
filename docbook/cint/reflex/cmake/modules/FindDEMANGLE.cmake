# - Find demangle functions
# This module finds the demangle library
#
# It sets the following variables:
#  DEMANGLE_FOUND       - Set to false, or undefined, if the demangle library isn't found.
#  DEMANGLE_LIBRARY     - The demangle library to link against.

INCLUDE(CheckFunctionExists)

FIND_LIBRARY(DEMANGLE_LIBRARY NAMES demangle PATHS /lib /usr/lib)

IF (DEMANGLE_LIBRARY)
   SET(DEMANGLE_FOUND TRUE)
ENDIF (DEMANGLE_LIBRARY)

IF (DEMANGLE_FOUND)
   IF (NOT DEMANGLE_FIND_QUIETLY)
      MESSAGE(STATUS "Found demangle: ${DEMANGLE_LIBRARY}")
   ENDIF (NOT DEMANGLE_FIND_QUIETLY)
ELSE (DEMANGLE_FOUND)
   IF (DEMANGLE_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find demangle")
   ENDIF (DEMANGLE_FIND_REQUIRED)
ENDIF (DEMANGLE_FOUND)
