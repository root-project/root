# genreflex crashes with relational operator in template argument
MACRO (test001)

   SET(_src "template<typename T>
             struct integer_traits
             {
                static const int const_min = 10;
             };

             template< int Category > struct int_least_helper {};

             template< long MinValue > // minimum value to require support
             struct int_min_value_t
             {
                typedef int_least_helper
                <
                (MinValue >= integer_traits<long>::const_min)
                > least;
             };")
   REFLEX_TEST_FILE(WRITE X.h "${_src}")

   REFLEX_ASSERT_GENREFLEX_CLI(ARGS X.h --quiet
                               RESULT EQUAL 0
                               STDOUT MATCHES "^$"
                               STDERR MATCHES "^$")

   MACRO_ASSERT_TEST_FILE_EXISTS("X_rflx.cpp")

ENDMACRO (test001)
REFLEX_ADD_MACRO_TEST(test001)
