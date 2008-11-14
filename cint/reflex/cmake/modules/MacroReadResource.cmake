# MACRO_READ_RESOURCE(_var_name _list_filename _name)
#   This macro will read the contents of a resource relative to the
#   specified file's directory and store it into the specified variable

MACRO (MACRO_READ_RESOURCE _var_name _list_filename _name)

   MACRO_GET_RESOURCE_FILENAME(_rc_filename ${_list_filename} ${_name})
   FILE(READ ${_rc_filename} ${_var_name})

ENDMACRO (MACRO_READ_RESOURCE _var_name _list_filename _name)
