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
    cling-parsing-semicolon
    cling-operator-ConversionOp
)
