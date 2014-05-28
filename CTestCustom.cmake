#-------------------------------------------------------------------------------
#
# CTestCustom.cmake
#
# This file enables customization of CTest.
#
#-------------------------------------------------------------------------------

# Specify tests that will be ignored.

set(CTEST_CUSTOM_TESTS_IGNORE
    ${CTEST_CUSTOM_TESTS_IGNORE}
    roottest-cling-parsing-semicolon
)
