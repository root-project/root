# MACRO_GET_TARGET_DIRECTORY(target _location)
#   This macro gets the absolute directory of a target's final location

MACRO(MACRO_GET_TARGET_DIRECTORY target _location)

   GET_TARGET_PROPERTY(${_location} ${target} LOCATION)
   GET_FILENAME_COMPONENT(${_location} ${${_location}} ABSOLUTE)
   GET_FILENAME_COMPONENT(${_location} ${${_location}} PATH)

ENDMACRO(MACRO_GET_TARGET_DIRECTORY target _location)
