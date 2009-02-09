# CLI Tests
#   Tests genreflex's command-line interface

REFLEX_RESOURCE_FILE(READ Synopsis.txt _msg_synopsis)
REFLEX_RESOURCE_FILE(READ Help.txt _msg_help)

# no input files is an error
MACRO (test001)

   REFLEX_ASSERT_GENREFLEX_CLI(RESULT EQUAL 1
                               STDOUT STREQUAL "--->> genreflex: ERROR: No input file specified\n${_msg_synopsis}"
                               STDERR MATCHES "^$")

ENDMACRO (test001)
REFLEX_ADD_MACRO_TEST(test001)


# -h prints the usage text on stdout
MACRO (test002)

   REFLEX_ASSERT_GENREFLEX_CLI(ARGS -h
                               RESULT EQUAL 0
                               STDOUT STREQUAL "${_msg_help}"
                               STDERR MATCHES "^$")

ENDMACRO (test002)
REFLEX_ADD_MACRO_TEST(test002)
