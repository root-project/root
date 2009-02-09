# MACRO (MACRO_JOIN_ARGUMENTS _var)
#   This macro joins the elements of the list elements into a space
#   separated string and stores the result in _var

MACRO (MACRO_JOIN_ARGUMENTS _var)

   SET(_joined)

   FOREACH (_arg ${ARGN})

      IF (_joined)
         SET(_joined "${_joined} ")
      ENDIF (_joined)
      SET(_joined "${_joined}${_arg}")

   ENDFOREACH (_arg ${ARGN})

   SET(${_var} "${_joined}")

ENDMACRO (MACRO_JOIN_ARGUMENTS _var)
