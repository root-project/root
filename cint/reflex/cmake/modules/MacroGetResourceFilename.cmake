# MACRO_GET_RESOURCE_FILENAME(_variable _list_filename _name)
#   This macro gets the absolute filename of a resource relative to the
#   specified file's directory

MACRO (MACRO_GET_RESOURCE_FILENAME _variable _list_filename _name)

   GET_FILENAME_COMPONENT(_absolute_dir ${_list_filename} PATH)
   SET(_rc_filename ${_absolute_dir}/${_name})
   FILE(TO_CMAKE_PATH _rc_filename ${_rc_filename})
   SET(${_variable} ${_rc_filename})

ENDMACRO (MACRO_GET_RESOURCE_FILENAME _variable _list_filename _name)
