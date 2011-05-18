# Tests for shadow classes used to calculate offsets
# even for non-public members / bases.

MACRO (FinalOverrider)

   REFLEX_ASSERT_GENREFLEX_CLI(FILES FinalOverrider.hpp
                               ARGS FinalOverrider.hpp --quiet
                               RESULT EQUAL 0
                               STDOUT MATCHES ".*"
                               STDERR MATCHES "^$")

   MACRO_ASSERT_TEST_FILE_EXISTS("FinalOverrider_rflx.cpp")

ENDMACRO (FinalOverrider)
REFLEX_ADD_MACRO_TEST(FinalOverrider)
