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
    roottest-root-meta-loadAllLibs-LoadAllLibs
    roottest-root-meta-loadAllLibs-LoadAllLibsAZ
    roottest-root-meta-loadAllLibs-LoadAllLibsZA
    roottest-root-html-runMakeIndex
)
