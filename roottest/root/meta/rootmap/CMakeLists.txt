ROOTTEST_COMPILE_MACRO(spaces.C
                       FIXTURES_SETUP root-meta-rootmap-spaces-fixture)

ROOTTEST_ADD_TEST(runspaces
                  MACRO runspaces.C
                  COPY_TO_BUILDDIR spaces.rootmap
                  OUTREF spaces.ref
                  FIXTURES_REQUIRED root-meta-rootmap-spaces-fixture)

ROOTTEST_ADD_TESTDIRS()
