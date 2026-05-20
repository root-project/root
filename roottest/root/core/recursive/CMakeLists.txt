ROOTTEST_GENERATE_EXECUTABLE(recursivewarning
                             recursivewarning.cxx
                             LIBRARIES Core
                             FIXTURES_SETUP root-core-recursive-recursivewarning-fixture)

ROOTTEST_ADD_TEST(recursivewarning
                  EXEC ./recursivewarning
                  OUTREF test_recursivewarning.ref
                  COPY_TO_BUILDDIR fakelib.rootmap
                  FIXTURES_REQUIRED root-core-recursive-recursivewarning-fixture)

