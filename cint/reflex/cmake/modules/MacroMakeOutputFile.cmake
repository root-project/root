# MACRO_MAKE_OUTPUT_FILE(infile prefix suffix ext outfile)
#   This macro calculates the file name for an output file in the build dir

MACRO (MACRO_MAKE_OUTPUT_FILE infile prefix suffix ext outfile)

   STRING(LENGTH ${CMAKE_CURRENT_BINARY_DIR} _binlength)
   STRING(LENGTH ${infile} _infileLength)

   SET(_checkinfile ${CMAKE_CURRENT_SOURCE_DIR})
   IF (_infileLength GREATER _binlength)
      STRING(SUBSTRING "${infile}" 0 ${_binlength} _checkinfile)
   ENDIF (_infileLength GREATER _binlength)

   IF (CMAKE_CURRENT_BINARY_DIR MATCHES "${_checkinfile}")
      FILE(RELATIVE_PATH rel ${CMAKE_CURRENT_BINARY_DIR} ${infile})
   ELSE (CMAKE_CURRENT_BINARY_DIR MATCHES "${_checkinfile}")
      FILE(RELATIVE_PATH rel ${CMAKE_CURRENT_SOURCE_DIR} ${infile})
   ENDIF (CMAKE_CURRENT_BINARY_DIR MATCHES "${_checkinfile}")

   SET(_outfile "${CMAKE_CURRENT_BINARY_DIR}/${rel}")
   GET_FILENAME_COMPONENT(outpath ${_outfile} PATH)
   GET_FILENAME_COMPONENT(_outfile ${_outfile} NAME_WE)
   FILE(MAKE_DIRECTORY ${outpath})
   SET(${outfile} ${outpath}/${prefix}${_outfile}${suffix}.${ext})

ENDMACRO (MACRO_MAKE_OUTPUT_FILE infile prefix suffix ext outfile)
