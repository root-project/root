# Error Handling Tests
#   Tests genreflex's handling of files with errors/warnings

# test that an output file is generated for valid input
MACRO (test001)

   REFLEX_TEST_FILE(WRITE X.h "class Y { };\n")

   REFLEX_ASSERT_GENREFLEX_CLI(ARGS X.h --quiet
                               RESULT EQUAL 0
                               STDOUT MATCHES ".*"
                               STDERR MATCHES "^$")

   MACRO_ASSERT_TEST_FILE_EXISTS("X_rflx.cpp")

ENDMACRO (test001)
REFLEX_ADD_MACRO_TEST(test001)


# test that an output file is not-generated ONLY for files containing errors
MACRO (test002)

   REFLEX_TEST_FILE(WRITE X.h "class X {int };\n")
   REFLEX_TEST_FILE(WRITE Y.h "class Y { };\n")

   REFLEX_ASSERT_GENREFLEX_CLI(ARGS X.h Y.h --quiet
                               RESULT EQUAL 1
                               STDOUT STREQUAL "\n--->> genreflex: ERROR: processing file with gccxml. genreflex command failed.\n"
                               STDERR MATCHES "X.h:1: error: .*\n")

   MACRO_ASSERT_TEST_FILE_NOT_EXISTS("X_rflx.cpp")
   MACRO_ASSERT_TEST_FILE_EXISTS("Y_rflx.cpp")

ENDMACRO (test002)
REFLEX_ADD_MACRO_TEST(test002)


# test that an output files are generated even if there are warnings
MACRO (test003)

   REFLEX_TEST_FILE(WRITE X.h "class X {int &y; };\n")
   REFLEX_TEST_FILE(WRITE Y.h "class Y { };\n")

   REFLEX_ASSERT_GENREFLEX_CLI(ARGS X.h Y.h --quiet
                               RESULT EQUAL 0
                               STDOUT STREQUAL "--->> genreflex: WARNING: References are not supported as data members (int& X::y)\n"
                               STDERR MATCHES "^$")

   MACRO_ASSERT_TEST_FILE_EXISTS("X_rflx.cpp")
   MACRO_ASSERT_TEST_FILE_EXISTS("Y_rflx.cpp")

ENDMACRO (test003)
REFLEX_ADD_MACRO_TEST(test003)


# With --fail_on_warnings, test that an output file is not-generated for all files (not just those containing warnings).
# to be more usable in Makefiles.
MACRO (test004)

   REFLEX_TEST_FILE(WRITE X.h "class X {int &y; };\n")
   REFLEX_TEST_FILE(WRITE Y.h "class Y { };\n")

   REFLEX_ASSERT_GENREFLEX_CLI(ARGS X.h Y.h --quiet --fail_on_warnings
                               RESULT EQUAL 1
                               STDOUT STREQUAL "--->> genreflex: WARNING: References are not supported as data members (int& X::y)\n--->> genreflex: ERROR: Exiting with error due to 1 warnings ( --fail_on_warnings enabled )\n"
                               STDERR MATCHES "^$")

   MACRO_ASSERT_TEST_FILE_NOT_EXISTS("X_rflx.cpp")
   MACRO_ASSERT_TEST_FILE_NOT_EXISTS("Y_rflx.cpp")
   MACRO_ASSERT_TEST_FILE_NOT_EXISTS("B.xml")

ENDMACRO (test004)
REFLEX_ADD_MACRO_TEST(test004)


# make sure that genreflex doesn't crash if gccxml fails
MACRO (test005)

   REFLEX_TEST_FILE(WRITE X.h "class X {int ; };\n")

   REFLEX_ASSERT_GENREFLEX_CLI(ARGS X.h
                               RESULT EQUAL 1
                               STDOUT STREQUAL "--->> genreflex: INFO: Parsing file X.h with GCC_XML \n--->> genreflex: ERROR: processing file with gccxml. genreflex command failed.\n"
                               STDERR STREQUAL "X.h:1: error: declaration does not declare anything\n")

   MACRO_ASSERT_TEST_FILE_NOT_EXISTS("X_rflx.cpp")

ENDMACRO (test005)
REFLEX_ADD_MACRO_TEST(test005)
