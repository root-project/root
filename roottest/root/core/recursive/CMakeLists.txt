ROOTTEST_GENERATE_EXECUTABLE(recursivewarning recursivewarning.cxx LIBRARIES Core)
ROOTTEST_ADD_TEST(recursivewarning
                  EXEC ./recursivewarning
                  OUTREF test_recursivewarning.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

configure_file(fakelib.rootmap . COPYONLY)

