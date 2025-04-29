ROOTTEST_ADD_AUTOMACROS(DEPENDS namingMatches.cxx
                        EXCLUDE execCheckNaming)

ROOTTEST_ADD_TEST(execCheckNaming
                  MACRO execCheckNaming.C
                  OUTREF execCheckNaming${ref_suffix})

ROOTTEST_ADD_TESTDIRS()
