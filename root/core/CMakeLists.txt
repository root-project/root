ROOTTEST_ADD_TEST(TEnv
                  MACRO  testTEnv.C
                  OUTREF testTEnv.ref)

ROOTTEST_ADD_TEST(assertHumanReadable
                  MACRO assertHumanReadable.cxx+
                  OUTREF assertHumanReadable.ref
                  )