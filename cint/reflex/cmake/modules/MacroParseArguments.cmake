# MACRO (MACRO_PARSE_ARGUMENTS prefix arg_names option_names)
#
#   From http://www.cmake.org/Wiki/CMakeMacroParseArguments:
#
#   The MACRO_PARSE_ARGUMENTS macro will take the arguments of another macro and
#   define several variables:
#
#     1) The first argument to is a prefix to put on all variables it creates.
#     2) The second argument is a quoted list of names,
#     3) and the third argument is a quoted list of options.
#
#   The rest of MACRO_PARSE_ARGUMENTS are arguments from another macro to be
#   parsed.
#
#   MACRO_PARSE_ARGUMENTS(prefix arg_names options arg1 arg2...)
#
#   For each item in options, MACRO_PARSE_ARGUMENTS will create a variable
#   with that name, prefixed with prefix_. So, for example, if prefix is
#   MY_MACRO and options is OPTION1;OPTION2, then PARSE_ARGUMENTS will create
#   the variables MY_MACRO_OPTION1 and MY_MACRO_OPTION2. These variables will
#   be set to true if the option exists in the command line or false otherwise.
#
#   For each item in arg_names, MACRO_PARSE_ARGUMENTS will create a variable
#   with that name, prefixed with prefix_. Each variable will be filled with the
#   arguments that occur after the given arg_name is encountered up to the next
#   arg_name or the end of the arguments. All options are removed from these
#   lists.
#
#   MACRO_PARSE_ARGUMENTS also creates a prefix_DEFAULT_ARGS variable containing
#   the list of all arguments up to the first arg_name encountered.

MACRO (MACRO_PARSE_ARGUMENTS prefix arg_names option_names)

   SET(DEFAULT_ARGS)

   FOREACH (arg_name ${arg_names})
      SET(${prefix}_${arg_name})
   ENDFOREACH (arg_name)

   FOREACH (option ${option_names})
      SET(${prefix}_${option} FALSE)
   ENDFOREACH (option)

   SET(current_arg_name DEFAULT_ARGS)
   SET(current_arg_list)

   FOREACH (arg ${ARGN})

      SET(larg_names ${arg_names})
      LIST(FIND larg_names "${arg}" is_arg_name)

      IF (is_arg_name GREATER -1)

         SET(${prefix}_${current_arg_name} ${current_arg_list})
         SET(current_arg_name "${arg}")
         SET(current_arg_list)

      ELSE (is_arg_name GREATER -1)

         SET(loption_names ${option_names})
         LIST(FIND loption_names "${arg}" is_option)

         IF (is_option GREATER -1)
            SET(${prefix}_${arg} TRUE)
         ELSE (is_option GREATER -1)
            SET(current_arg_list ${current_arg_list} "${arg}")
         ENDIF (is_option GREATER -1)

      ENDIF (is_arg_name GREATER -1)

   ENDFOREACH (arg ${ARGN})

   SET(${prefix}_${current_arg_name} ${current_arg_list})

ENDMACRO (MACRO_PARSE_ARGUMENTS)
