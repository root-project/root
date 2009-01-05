# MACRO (PYTHON_COMPILE _OUTFILES_VAR file1 ... fileN)
#
#   Compiles python sources to the build directory.


MACRO (PYTHON_COMPILE outfiles)

   SET(infiles ${ARGN})
   SET(_outfiles)

   FOREACH (f ${infiles})
   
      GET_FILENAME_COMPONENT(_in ${f} ABSOLUTE)
      MACRO_MAKE_OUTPUT_FILE(${_in} "" "" pyc _out)
   
      GET_FILENAME_COMPONENT(_out_dir ${_out} PATH)
      FILE(RELATIVE_PATH _out_rel ${CMAKE_BINARY_DIR} ${_out})
   
      ADD_CUSTOM_COMMAND(OUTPUT ${_out}
                         COMMAND ${PYTHON_EXECUTABLE}
                         ARGS -c "import py_compile; py_compile.compile('${_in}', '${_out}')"
                         DEPENDS ${_in}
                         COMMENT "Building Python object ${_out_rel}"
                         VERBATIM)
   
      LIST(APPEND _outfiles ${_out})
   
   ENDFOREACH (f ${infiles})
   
   # mark the compiled python objects as generated
   SET_SOURCE_FILES_PROPERTIES(${_outfiles} PROPERTIES GENERATED 1)
   
   # mark the compiled python objects for cleaning
   MACRO_ADDITIONAL_CLEAN_FILES(${_outfiles})

   SET(${outfiles} ${_outfiles})

ENDMACRO (PYTHON_COMPILE outfiles)
