ROOTTEST_ADD_TEST(runspaces
                  MACRO runspaces.C
                  OUTREF spaces.ref
                  DEPENDS spaces.C)

ROOTTEST_ADD_TESTDIRS()
