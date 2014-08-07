#---Custom CTest settings---------------------------------------------------

set(CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE "100000")
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE "10000")

set(CTEST_TEST_TIMEOUT 1200)

set(CTEST_CUSTOM_WARNING_EXCEPTION ${CTEST_CUSTOM_WARNING_EXCEPTION}
        "Warning: Rank mismatch in argument"
        "Warning: Actual argument contains too few elements"
        "has no symbols"                                         # library.a(object.c.o) has no symbols
        "note: variable tracking size limit exceeded"            # vc/tests/sse_blend.cpp
        "warning is a GCC extension"
        "bug in GCC 4.8.1"
        "warning: please use fgets or getline instead"           # deprecated use of std functions cint/ROOT
        "is dangerous, better use"                               # deprecated use of std functions cint/ROOT
        "function is dangerous and should not be used"           # deprecated use of std functions cint/ROOT
    )
set(CTEST_CUSTOM_ERROR_EXCEPTION ${CTEST_CUSTOM_ERROR_EXCEPTION}
        "fatal error: cannot open file")

#---Include other CTest Custom files----------------------------------------
include(test/CTestCustom.cmake OPTIONAL)
include(roottest/CTestCustom.cmake OPTIONAL)
include(tutorials/CTestCustom.cmake OPTIONAL)
