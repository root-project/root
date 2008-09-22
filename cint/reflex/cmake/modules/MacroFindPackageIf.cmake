# MACRO_FIND_PACKAGE_IF(_name _cond)
#   This macro works like FIND_PACKAGE() if the condition evaluates to
#   true. The standard <name>_FOUND variables can be used in the same way
#   as when using the normal FIND_PACKAGE().
#   Inspired by KDE's MACRO_OPTIONAL_FIND_PACKAGE macro

MACRO (MACRO_FIND_PACKAGE_IF _name _cond)

   IF (${_cond})
      FIND_PACKAGE(${_name} ${ARGN})
   ELSE (${_cond})
      SET(${_name}_FOUND)
      SET(${_name}_INCLUDE_DIR)
      SET(${_name}_INCLUDES)
      SET(${_name}_LIBRARY)
      SET(${_name}_LIBRARIES)
   ENDIF (${_cond})

ENDMACRO (MACRO_FIND_PACKAGE_IF _name _cond)
