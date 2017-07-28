ROOTTEST_ADD_TEST(TEnv
                  MACRO  testTEnv.C
                  OUTREF testTEnv.ref)

ROOTTEST_ADD_TEST(assertHumanReadable
                  MACRO assertHumanReadable.cxx+
                  OUTREF assertHumanReadable.ref
                  )
ROOTTEST_ADD_TEST(assertROOT7244
                  MACRO assertROOT7244.C
                  COPY_TO_BUILDDIR subdir_ROOT7244/TheFile.h 
                  )
ROOTTEST_ADD_TEST(assertROOT8542
                  MACRO assertROOT8542.C
                  )
